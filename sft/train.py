import os
import sys
import torch.amp
import torch.distributed as dist
from accelerate import DistributedDataParallelKwargs, Accelerator
from loguru import logger
from tensorboardX import SummaryWriter
from transformers import get_scheduler
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig, TaskType
from peft.tuners.lora import LoraLayer

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sft.dataloader import infinite_dataloader
from sft.dataset import build_dataset
from sft.utils import load_yaml_cfg, send_to_device, set_module_train_state


class Trainer:
    def __init__(self, train_config: str) -> None:
        self.train_config = train_config
        self.config = load_yaml_cfg(train_config)
        self.output_dir = self.config['train']['save_config']['save_dir']
        self.init_env()
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            self.config['train']['model']['pretrained_model_path']
        )
        use_grouped_gemm = self.config['train']['model']['use_grouped_gemm']
        if use_grouped_gemm and not self.model.config.llm_config.use_grouped_gemm:
            self.model.model.fuse_experts()

        set_module_train_state(self.model.audio, freeze=True)
        if not self.config['train']['model']['freeze_semantic_module']:
            set_module_train_state(self.model.audio.decoder.fc2, freeze=False)
            set_module_train_state(self.model.audio.decoder.semantic_model, freeze=False)

        if self.config['train']['model']['sft_type'] == 'lora':
            assert not use_grouped_gemm, "The LoRA training does not support grouped_gemm for now"
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

        self.dataset = build_dataset(**self.config['dataset'])

        self.dataloader = infinite_dataloader(
            self.dataset,
            **self.config['train']['dataloader']
        )

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            mixed_precision='bf16',
            gradient_accumulation_steps=1,
            kwargs_handlers=[ddp_kwargs]
        )
        
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.config['train']['optimizer']
        )
    
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        self.lr_scheduler = get_scheduler(
            optimizer=self.optimizer,
            **self.config['train']['lr_scheduler']
        )
        self.model.train()

    def init_env(self):
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        if dist.get_rank() == 0:
            os.makedirs(self.output_dir, exist_ok=True)
            if os.path.exists(f'{self.output_dir}/train.log'):
                os.system(f"rm {self.output_dir}/train.log")
            logger.add(sink=f'{self.output_dir}/train.log')
            self.summary_writer = SummaryWriter(log_dir=f'{self.output_dir}/tfevent')

        logger.info(f'{rank}-{local_rank}/{world_size}')

    def train(self):
        steps = 0
        stats_duration = 0.0

        flow_loss_w = self.config['train']['loss_weight']['flow_loss']
        stop_loss_w = self.config['train']['loss_weight']['stop_loss']
        asr_loss_w = self.config['train']['loss_weight']['asr_loss']

        # training task type
        if asr_loss_w == 0:
            task_type = 'tts'
        else:
            task_type = 'asr'

        for batch in self.dataloader:
            batch = send_to_device(batch, 'cuda')
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