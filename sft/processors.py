import math
import torch
import torchaudio
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from loguru import logger


class SampleBuilder:
    def __init__(self, tokenizer, sr=16000, patch_size=5, hop_size=320):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        self.sr = sr
        self.patch_size = patch_size
        self.hop_size = hop_size
    
    def build_sample(self, sample):
        if sample['task_type'] == 'tts':
            sample = self.build_tts_sample(sample)
        elif sample['task_type'] == 'asr':
            sample = self.build_asr_sample(sample)
        return sample
    
    def build_tts_sample(self, sample):
        if 'waveform' in sample and 'sample_rate' in sample:
            waveform, sample_rate = sample['waveform'], sample['sample_rate']
        else:
            waveform, sample_rate = torchaudio.load(sample['wav_path'], backend='soundfile')
        if waveform.ndim == 2:
            waveform = waveform[0]

        if sample_rate != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)
        
        input_part = (
            self.tokenizer.encode("<role>HUMAN</role>") +
            self.tokenizer.encode(f"Please translate the text to speech.\n{sample['text']}") +
            self.tokenizer.encode("<role>ASSISTANT</role>")
        )
        
        length = math.ceil(len(waveform) / self.hop_size) // self.patch_size
        output_part = [self.tokenizer.convert_tokens_to_ids("<audio>")] + [self.tokenizer.convert_tokens_to_ids('<audioPatch>')] * length
        placeholder_loc_lens = torch.tensor([(len(input_part)+1, length)], dtype=torch.long)
        
        input_ids = torch.tensor(input_part+output_part, dtype=torch.int)
        attention_mask = torch.ones_like(input_ids)
        stop_label = torch.full_like(attention_mask, fill_value=-100, dtype=torch.long)
        stop_label[placeholder_loc_lens[0][0]:placeholder_loc_lens[0][0]+placeholder_loc_lens[0][1]-4] = 0
        stop_label[placeholder_loc_lens[0][0]+placeholder_loc_lens[0][1]-1] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'waveform': waveform,
            'waveform_length':  torch.tensor([len(waveform)], dtype=torch.long),
            'stop_label': stop_label,
            'placeholder_loc_lens': placeholder_loc_lens,
            'stats_duration': torch.tensor([len(waveform)/self.sr/3600], dtype=torch.float32)
        }

    def build_asr_sample(self, sample):
        if 'waveform' in sample and 'sample_rate' in sample:
            waveform, sample_rate = sample['waveform'], sample['sample_rate']
        else:
            waveform, sample_rate = torchaudio.load(sample['wav_path'], backend='soundfile')
        if waveform.ndim == 2:
            waveform = waveform[0]

        if sample_rate != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sr)(waveform)

        length = math.ceil(len(waveform) / self.hop_size) // self.patch_size

        prompt_tokens = self.tokenizer.encode("<role>HUMAN</role>Please recognize the language of this speech and transcribe it. Format: oral.")
        input_part = (
            prompt_tokens +
            self.tokenizer.encode(f"<audio>") +
            self.tokenizer.encode(f"<audioPatch>")*length +
            self.tokenizer.encode(f"</audio>") +
            self.tokenizer.encode(f"<role>ASSISTANT</role>")
        )

        output_part = self.tokenizer.encode(f"{sample['lang']}\t{sample['text']}<|endoftext|>")
        placeholder_loc_lens = torch.tensor([(len(prompt_tokens)+1, length)], dtype=torch.long)

        input_ids = torch.tensor(input_part+output_part, dtype=torch.int)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([-100]*len(input_part) + output_part, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'waveform': waveform,
            'waveform_length':  torch.tensor([len(waveform)], dtype=torch.long),
            'labels': labels,
            'placeholder_loc_lens': placeholder_loc_lens,
            'stats_duration': torch.tensor([len(waveform)/self.sr/3600], dtype=torch.float32)
        }


    def __call__(self, data):
        for sample in data:
            yield self.build_sample(sample)


class DynamicBatch:
    def __init__(self, max_frames_in_batch=6000):
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, data):
        buf = []
        longest_frames = 0
        for sample in data:
            new_sample_frames = len(sample['input_ids'])
            longest_frames = max(longest_frames, new_sample_frames)

            frames_after_padding = longest_frames * (len(buf) + 1)
            if frames_after_padding > self.max_frames_in_batch:
                buf.sort(key=lambda x: len(x['input_ids']))
                yield buf
                buf = [sample]
                longest_frames = new_sample_frames
            else:
                buf.append(sample)
        if len(buf) > 0:
            yield buf


class Sort:
    def __init__(self, buffer_size=1000):
        self.buffer_size = buffer_size

    def __call__(self, data):
        buf = []
        tmp_buf = []
        iterator = iter(data)

        for _ in range(self.buffer_size):
            try:
                sample = next(iterator)
                buf.append(sample)
            except StopIteration:
                break

        buf.sort(key=lambda x: len(x['input_ids']))

        try:
            while True:
                sample = next(iterator)
                if sample is None:
                    continue
                if len(buf) == 0:
                    buf = tmp_buf
                    buf.sort(key=lambda x: len(x['input_ids']))
                    tmp_buf = []
                if len(buf) > 0:
                    yield buf.pop()
                tmp_buf.append(sample)
        except StopIteration:
            logger.debug("Processor `Sort` triggers StopIteration. This is EXPECTED.")

        buf.extend(tmp_buf)
        if len(buf) > 0:
            buf.sort(key=lambda x: len(x['input_ids']))
            for x in buf:
                yield x


class Padding:
    def __init__(self, tokenizer):
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        else:
            self.tokenizer = tokenizer
        
        self.pad_val_map = {
            'input_ids': self.tokenizer.pad_token_id,
            'labels': -100,
            'stop_label': -100,
        }

    def __call__(self, data):
        for batch in data:
            new_batch = {}

            for key in batch[0].keys():
                new_batch[key] = pad_sequence([x[key] for x in batch], batch_first=True, padding_value=self.pad_val_map.get(key, 0))

            for key in ['waveform_length', 'stats_duration']:
                if key in new_batch:
                    new_batch[key] = new_batch[key].squeeze(-1)

            yield new_batch