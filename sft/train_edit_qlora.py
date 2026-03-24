import os
import sys
import torch
import torch.amp
import torch.distributed as dist
from accelerate import DistributedDataParallelKwargs, Accelerator
from loguru import logger
from tensorboardX import SummaryWriter
from transformers import get_scheduler, BitsAndBytesConfig
import bitsandbytes as bnb
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer

import math
import torchaudio
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration, BailingMMCausalLMOutputWithPast
from sft.dataloader import infinite_dataloader
from sft.dataset import build_dataset, SftDataset, Processor
from sft.processors import SampleBuilder, Padding, DynamicBatch, Sort
from sft.utils import load_yaml_cfg, send_to_device, set_module_train_state, length2attention_mask

class EditSampleBuilder(SampleBuilder):
    def build_sample(self, sample):
        if sample['task_type'] == 'edit':
            return self.build_edit_sample(sample)
        return super().build_sample(sample)
        
    def build_edit_sample(self, sample):
        if 'waveform' in sample and 'sample_rate' in sample:
            raise NotImplementedError("Edit task doesn't support pre-loaded waveforms.")
            
        prompt_waveform, sr1 = torchaudio.load(sample['prompt_wav_path'], backend='soundfile')
        target_waveform, sr2 = torchaudio.load(sample['target_wav_path'], backend='soundfile')
        
        if prompt_waveform.ndim == 2: prompt_waveform = prompt_waveform[0]
        if target_waveform.ndim == 2: target_waveform = target_waveform[0]
            
        if sr1 != self.sr: prompt_waveform = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=self.sr)(prompt_waveform)
        if sr2 != self.sr: target_waveform = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=self.sr)(target_waveform)
            
        prompt_length = math.ceil(len(prompt_waveform) / self.hop_size) // self.patch_size
        target_length = math.ceil(len(target_waveform) / self.hop_size) // self.patch_size
        
        input_part = (
            self.tokenizer.encode("<role>HUMAN</role>") +
            [self.tokenizer.convert_tokens_to_ids("<audio>")] + 
            [self.tokenizer.convert_tokens_to_ids('<audioPatch>')] * prompt_length +
            self.tokenizer.encode(f"<prompt>{sample['text']}\n</prompt>") +
            self.tokenizer.encode("<role>ASSISTANT</role>") +
            self.tokenizer.encode("<gen_audio>")
        )
        
        output_part_audio = [self.tokenizer.convert_tokens_to_ids("<audio>")] + [self.tokenizer.convert_tokens_to_ids('<audioPatch>')] * target_length
        
        input_ids = torch.tensor(input_part + output_part_audio, dtype=torch.int)
        attention_mask = torch.ones_like(input_ids)
        
        audio_placeholder_loc_lens = torch.tensor([
            (1, prompt_length), 
            (len(input_part) + 1, target_length)
        ], dtype=torch.long)
        
        combined_waveform = torch.cat([prompt_waveform, target_waveform], dim=-1)
        encoded_feat_lens = torch.tensor([prompt_length, target_length], dtype=torch.long)
        
        stop_label = torch.full_like(attention_mask, fill_value=-100, dtype=torch.long)
        target_start = audio_placeholder_loc_lens[1][0]
        stop_label[target_start : target_start + target_length - 4] = 0
        stop_label[target_start + target_length - 1] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'waveform': combined_waveform,
            'waveform_length': torch.tensor([len(combined_waveform)], dtype=torch.long),
            'stop_label': stop_label,
            'placeholder_loc_lens': audio_placeholder_loc_lens,
            'encoded_feat_lens': encoded_feat_lens,
            'stats_duration': torch.tensor([len(combined_waveform)/self.sr/3600], dtype=torch.float32)
        }

class EditPadding(Padding):
    def __call__(self, data):
        for batch in data:
            new_batch = {}
            for key in batch[0].keys():
                if key == 'placeholder_loc_lens':
                    max_len = max([x[key].shape[0] for x in batch])
                    padded_tensors = []
                    for x in batch:
                        if x[key].shape[0] < max_len:
                            pad = torch.zeros(max_len - x[key].shape[0], 2, dtype=x[key].dtype)
                            padded_tensors.append(torch.cat([x[key], pad], dim=0))
                        else:
                            padded_tensors.append(x[key])
                    new_batch[key] = torch.stack(padded_tensors, dim=0)
                elif key == 'encoded_feat_lens':
                    max_len = max([x[key].shape[0] for x in batch])
                    padded_tensors = []
                    for x in batch:
                        if x[key].shape[0] < max_len:
                            pad = torch.zeros(max_len - x[key].shape[0], dtype=x[key].dtype)
                            padded_tensors.append(torch.cat([x[key], pad], dim=0))
                        else:
                            padded_tensors.append(x[key])
                    new_batch[key] = torch.stack(padded_tensors, dim=0)
                else:
                    new_batch[key] = pad_sequence([x[key] for x in batch], batch_first=True, padding_value=self.pad_val_map.get(key, 0))

            for key in ['waveform_length', 'stats_duration']:
                if key in new_batch:
                    new_batch[key] = new_batch[key].squeeze(-1)
            yield new_batch

def build_edit_dataset(data_jsonl_file, tokenizer=False, sr=16000, patch_size=5, hop_size=320, max_frames_in_batch=1000, buffer_size=100):
    dataset = SftDataset(data_jsonl_file)
    dataset = Processor(dataset, EditSampleBuilder(tokenizer, sr=sr, patch_size=patch_size, hop_size=hop_size))
    dataset = Processor(dataset, Sort(buffer_size=buffer_size))
    dataset = Processor(dataset, DynamicBatch(max_frames_in_batch=max_frames_in_batch))
    dataset = Processor(dataset, EditPadding(tokenizer=tokenizer))
    return dataset

class BailingMMEditModel(BailingMMNativeForConditionalGeneration):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        stop_label=None,
        waveform=None,
        waveform_length=None,
        placeholder_loc_lens=None,
        past_key_values=None,
        **kwargs
    ):
        task_type = kwargs.pop('task_type', 'edit')
        encoded_feat_lens = kwargs.pop('encoded_feat_lens', None)
        
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify input_ids or inputs_embeds")

        audio_embeds, audio_embeds_lengths, latents = self.extract_audio_feature(waveform, waveform_length)
        if encoded_feat_lens is not None:
            audio_embeds_lengths = encoded_feat_lens
            
        words_embeddings, audio_mask = self.prompt_wrap_navit(
            input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
            audio_embeds,
            audio_embeds_lengths,
            placeholder_loc_lens,
            None,
        )
        
        if self.config.llm_config.rope_scaling is not None and self.config.llm_config.rope_scaling["type"] == "3D":
             position_ids, rope_deltas = self.get_rope_index(
                 input_ids, image_token_id=self.config.llm_config.image_patch_token,
                 video_token_id=self.config.llm_config.image_patch_token,
                 image_start_token_id=self.config.llm_config.image_start_token,
                 video_start_token_id=self.config.llm_config.video_start_token,
                 image_grid_thw=None, video_grid_thw=None, attention_mask=attention_mask,
             )
             self.rope_deltas = rope_deltas
        else:
             rope_deltas = None

        outputs = self.model(
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=words_embeddings, labels=labels, use_cache=False, output_attentions=False,
            output_hidden_states=True, return_dict=True
        )
        llm_last_outputs = outputs.hidden_states[-1]
        
        if task_type == 'edit':
            B = audio_embeds.size(0)
            prompt_lengths = audio_embeds_lengths[:, 0]
            target_lengths = audio_embeds_lengths[:, 1]
            
            max_target_len = target_lengths.max().item()
            condition = torch.zeros(B, max_target_len, audio_embeds.size(-1)).to(audio_embeds.device)
            for i in range(B):
               target_start_loc = placeholder_loc_lens[i, 1, 0].item() - 1
               condition[i, :target_lengths[i]] = llm_last_outputs[i, target_start_loc : target_start_loc + target_lengths[i]]
               
            condition = condition.reshape(-1, 1, condition.size(-1))
            condition_mask = length2attention_mask(target_lengths).reshape(-1, 1).expand(-1, self.patch_size)
            
            target_latents = torch.zeros(B, max_target_len, latents.size(-1)).to(latents.device)
            for i in range(B):
               target_latents[i, :target_lengths[i]] = latents[i, prompt_lengths[i] : prompt_lengths[i] + target_lengths[i]]
               
            latent_history = target_latents.clone()[:, :-self.patch_size, :]
            latent_history = F.pad(latent_history, pad=(0, 0, self.patch_size, 0))
            latent_history = latent_history.reshape(-1, self.patch_size, latent_history.size(-1))
            target = target_latents.clone().reshape(-1, self.patch_size, target_latents.size(-1))

            flow_loss = self.flowloss(
                cond=condition, target=target, latent_history=latent_history,
                mask=condition_mask, patch_size=self.patch_size
            )

            pred_stop_probs = self.stop_head(llm_last_outputs).transpose(1, 2)
            stop_loss = self.stop_loss_func(pred_stop_probs, stop_label)
            asr_loss = torch.zeros_like(flow_loss)
            
        else:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, 
                                   inputs_embeds=inputs_embeds, labels=labels, stop_label=stop_label, waveform=waveform, 
                                   waveform_length=waveform_length, placeholder_loc_lens=placeholder_loc_lens, 
                                   past_key_values=past_key_values, task_type=task_type, **kwargs)

        return BailingMMCausalLMOutputWithPast(
            flow_loss=flow_loss, stop_loss=stop_loss[0], stop_loss_likely=stop_loss[1], asr_loss=asr_loss
        )

class Trainer:
    def __init__(self, train_config: str) -> None:
        self.train_config = train_config
        self.config = load_yaml_cfg(train_config)
        self.output_dir = self.config['train']['save_config']['save_dir']
        
        # Initialize accelerate first before loading model to get proper device
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=self.config['train'].get('gradient_accumulation_steps', 1),
            kwargs_handlers=[ddp_kwargs]
        )
        
        self.init_env()
        
        sft_type = self.config['train']['model'].get('sft_type', 'lora')
        use_grouped_gemm = self.config['train']['model'].get('use_grouped_gemm', False)
        
        print(f"Loading model with SFT Type: {sft_type}")
        
        if sft_type == 'qlora':
            assert not use_grouped_gemm, "QLoRA does not support grouped_gemm"
            # Define 4-bit quantization config (QLoRA)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.model = BailingMMEditModel.from_pretrained(
                self.config['train']['model']['pretrained_model_path'],
                quantization_config=bnb_config,
                device_map={"": self.accelerator.local_process_index}
            )
            # Prepare for k-bit training
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=True)
        else:
            self.model = BailingMMEditModel.from_pretrained(
                self.config['train']['model']['pretrained_model_path']
            )

        if use_grouped_gemm and not self.model.config.llm_config.use_grouped_gemm:
            self.model.model.fuse_experts()

        set_module_train_state(self.model.audio, freeze=True)
        if not self.config['train']['model']['freeze_semantic_module']:
            set_module_train_state(self.model.audio.decoder.fc2, freeze=False)
            set_module_train_state(self.model.audio.decoder.semantic_model, freeze=False)

        if sft_type in ['lora', 'qlora']:
            assert not use_grouped_gemm, "LoRA/QLoRA does not support grouped_gemm for now"
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                **self.config['train']['model']['lora_config']
            )
            self.model = get_peft_model(self.model, lora_config)

            # Don't apply LoRA to model.audio; revert these modules manually due to target_modules conflict
            for name, module in self.model.named_modules():
                if not isinstance(module, LoraLayer):
                    continue
                if "model.audio" in name:
                    setattr(
                        self.model.get_submodule(name.rsplit('.', 1)[0]),
                        name.rsplit('.', 1)[-1],
                        module.base_layer
                    )
            self.model.print_trainable_parameters()

        self.dataset = build_edit_dataset(**self.config['dataset'])

        self.dataloader = infinite_dataloader(
            self.dataset,
            **self.config['train']['dataloader']
        )
        
        # Use Paged AdamW 32-bit to offload optimizer states to CPU RAM (fits Kaggle's 30GB RAM/15GB VRAM well)
        self.optimizer = bnb.optim.PagedAdamW32bit(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config['train']['optimizer']
        )
    
        # If QLoRA is used, device_map already puts the model on the correct GPU, 
        # but accelerator still needs to wrap it for DDP.
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.lr_scheduler = get_scheduler(
            optimizer=self.optimizer,
            **self.config['train']['lr_scheduler']
        )
        self.model.train()

    def init_env(self):
        # We rely on Accelerator for process group initialization. 
        # But we setup logger for the main process.
        if int(os.environ.get('RANK', 0)) == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            if os.path.exists(f'{self.output_dir}/train.log'):
                os.system(f"rm {self.output_dir}/train.log")
            logger.add(sink=f'{self.output_dir}/train.log')
            self.summary_writer = SummaryWriter(log_dir=f'{self.output_dir}/tfevent')
            
            self.use_wandb = self.config.get('use_wandb', True)
            if self.use_wandb:
                import wandb
                wandb.init(
                    project=self.config.get('wandb_project', 'ming-audio-edit'),
                    name='qlora-edit-finetune',
                    config=self.config,
                    dir=self.output_dir,
                    resume="allow"
                )

        logger.info(f"{os.environ.get('RANK', 0)}-{os.environ.get('LOCAL_RANK', 0)}/{os.environ.get('WORLD_SIZE', 1)}")

    def train(self):
        steps = 0
        stats_duration = 0.0

        flow_loss_w = self.config['train']['loss_weight']['flow_loss']
        stop_loss_w = self.config['train']['loss_weight']['stop_loss']
        asr_loss_w = self.config['train']['loss_weight']['asr_loss']

        # training task type
        task_type = 'edit'

        for batch in self.dataloader:
            batch = send_to_device(batch, self.accelerator.device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                outputs = self.model(**batch, task_type=task_type)

            flow_loss = outputs.flow_loss
            stop_loss = outputs.stop_loss
            stop_loss_likely = outputs.stop_loss_likely
            asr_loss = outputs.asr_loss

            loss = flow_loss*flow_loss_w + stop_loss*stop_loss_w + asr_loss*asr_loss_w
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            steps += 1

            flow_loss_mean = self.accelerator.reduce(flow_loss, reduction='mean')
            stop_loss_mean = self.accelerator.reduce(stop_loss, reduction='mean')
            asr_loss_mean = self.accelerator.reduce(asr_loss, reduction='mean')
            
            stats_duration += self.accelerator.reduce(batch['stats_duration'].sum(), reduction='sum').item()

            if steps % 10 == 0 and self.accelerator.is_main_process:
                logger.info(
                    {
                        'step': steps,
                        'lr': self.lr_scheduler.get_lr(),
                        'flow_loss': flow_loss_mean.item(),
                        'stop_loss': stop_loss_mean.item(),
                        'stop_loss_likely': stop_loss_likely,
                        'asr_loss': asr_loss_mean.item(),
                        'duration_sum': stats_duration
                    }
                )

                self.summary_writer.add_scalar('loss/flow_loss', flow_loss_mean.item(), steps)
                self.summary_writer.add_scalar('loss/stop_loss', stop_loss_mean.item(), steps)
                self.summary_writer.add_scalar('loss/asr_loss', asr_loss_mean.item(), steps)
                self.summary_writer.add_scalar('lr', self.lr_scheduler.get_lr()[0], steps)
                self.summary_writer.add_scalar('stats/duration', stats_duration, steps)
                
                if getattr(self, 'use_wandb', True):
                    import wandb
                    wandb.log({
                        'step': steps,
                        'loss/flow_loss': flow_loss_mean.item(),
                        'loss/stop_loss': stop_loss_mean.item(),
                        'loss/asr_loss': asr_loss_mean.item(),
                        'lr': self.lr_scheduler.get_lr()[0],
                        'stats/duration': stats_duration
                    }, step=steps)

            if steps % self.config['train']['save_config']['save_model_steps'] == 0:
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                unwrapped_model.save_pretrained(
                    f'{self.output_dir}/ckpts/{steps}',
                    is_main_process=self.accelerator.is_main_process,
                    save_function=self.accelerator.save,
                    state_dict=self.accelerator.get_state_dict(self.model)
                )

            if steps == self.config['train']['lr_scheduler']['num_training_steps']:
                logger.info('Training finished!')
                break

if __name__ == '__main__':
    trainer = Trainer(sys.argv[1])
    trainer.train()
