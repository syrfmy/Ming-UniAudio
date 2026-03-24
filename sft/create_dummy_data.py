import json
import torch
import torchaudio
import os

# Create data directory
os.makedirs('sft/data/dummy', exist_ok=True)

# Generate small 1-second sine wave audios to act as placeholders
def generate_sine_wave(filepath, freq=440.0, sr=16000, duration=1.0):
    t = torch.linspace(0, duration, int(sr * duration))
    waveform = 0.5 * torch.sin(2 * torch.pi * freq * t).unsqueeze(0)
    torchaudio.save(filepath, waveform, sr)

# Create 10 dummy pairs for prompt and target
dummy_data = []
for i in range(10):
    prompt_path = f"sft/data/dummy/prompt_{i}.wav"
    target_path = f"sft/data/dummy/target_{i}.wav"
    
    generate_sine_wave(prompt_path, freq=440.0 + i*10)
    generate_sine_wave(target_path, freq=880.0 + i*10)
    
    dummy_data.append({
        "task_type": "edit",
        "prompt_wav_path": prompt_path,
        "target_wav_path": target_path,
        "text": f"This is a dummy test instruction number {i} for making editing work on Kaggle."
    })

# Write the JSONL
jsonl_path = "sft/data/my_custom_data.jsonl"
with open(jsonl_path, 'w') as f:
    for item in dummy_data:
        f.write(json.dumps(item) + '\n')

print(f"Created {len(dummy_data)} dummy audio pairs and saved dataset to {jsonl_path}")
