# Supervised Fine-Tuning (SFT) for Ming-UniAudio

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Speech Generation](#speech-generation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
- [Speech Understanding](#speech-understanding)
  - [Data Preparation](#data-preparation-1)
  - [Training](#training-1)
- [Acknowledgements](#acknowledgements)

## Overview

Ming-UniAudio is a unified model designed for both speech understanding and generation, including capabilities for speech generation, understanding, and editing. This open-source release focuses on the **SFT**. It is equipped with a robust set of features to support efficient and scalable training.

## Features
*   **Distributed Training**: Support for multi-node, multi-GPU training.
*   **FSDP Integration**: Natively supports PyTorch's FSDP (Fully Sharded Data Parallel) for training large models efficiently.
*   **Dynamic Batching**: Optimizes GPU utilization by handling sequences of varying lengths.
*   **Flexible Training Methods**:
    *   Full-parameter fine-tuning.
    *   LoRA (Low-Rank Adaptation) fine-tuning.
*   **Performance Optimization**: Includes support for `grouped_gemm` to significantly accelerate training.

## Installation

### 1. Main Dependencies
First, install the required packages from the root directory of the project:
```bash
pip install -r requirements.txt
```

### 2. Grouped GEMM for Performance Boost (Recommended)

For a significant performance increase, we highly recommend installing the `grouped_gemm` library. In our tests on H800 GPUs, this optimization provided a **~3x training speedup**.

We use the implementation from [fanshiqing/grouped_gemm](https://github.com/fanshiqing/grouped_gemm.git).

> **Important Note on PyTorch Version:** The official installation for `grouped_gemm` may attempt to install PyTorch 2.9, which has not been verified in our environment. The following instructions are tailored to compile `grouped_gemm` from source while reusing your existing PyTorch environment (e.g., v2.6).

Follow these steps to build and install it from source:
```bash
# Ensure the linker can find necessary CUDA and Conda libraries
export LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Clone the repository and its submodules
git clone https://github.com/fanshiqing/grouped_gemm.git
cd grouped_gemm
git submodule update --init --recursive

# Build and install from source
MAX_JOBS=16 python setup.py build_ext --force
python setup.py install --force
```

## Speech Generation
### Data Preparation

The training data must be in a **JSON Lines** (`.jsonl`) format. Each line in the file should be a JSON object containing the following keys:

*   `task_type`: The task identifier. For speech generation, this should be `"tts"`.
*   `wav_path`: The absolute or relative path to the audio file.
*   `text`: The corresponding text transcript for the audio.

**Example:**

Please refer to `sft/data/tts.jsonl` for a sample. A single line should look like this:

```json
{"task_type": "tts", "wav_path": "/path/to/your/audio/sample1.wav", "text": "This is the audio transcription."}
```

### Training

Ensure you run the training scripts from the **root directory** of the `Ming-Lite-UniAudio` project.

- Full-Parameter Fine-Tuning(Recommended)

To start a full-parameter SFT job, use the following command:

```bash
bash sft/train.sh sft/conf/train_tts.yaml
```

- LoRA Fine-Tuning

To start a LoRA-based SFT job, use the following command:

```bash
bash sft/train.sh sft/conf/train_tts_lora.yaml
```

You can customize training parameters, data paths, and model configurations by editing the corresponding `.yaml` files in the `sft/conf/` directory.

## Speech Understanding
### Data Preparation
The training data must be in a **JSON Lines** (`.jsonl`) format. Each line in the file should be a JSON object containing the following keys:

*   `task_type`: The task identifier. For speech generation, this should be `"asr"`.
*   `wav_path`: The absolute or relative path to the audio file.
*   `text`: The corresponding text transcript for the audio.
*   `lang`: The language/dialect type, supports `['Chinese', 'English', '川渝', '湖南', '闽南', '上海', 'Canton']`

**Example:**

Please refer to `sft/data/asr.jsonl` for a sample. A single line should look like this:
```json
{"task_type": "asr", "wav_path": "/input/lyuyongjie.lyj/data/testsets/aishell1/wav/test/S0915/BAC009S0915W0292.wav", "text": "现在是不是也该长点心了吧", "lang": "Chinese"}
```
### Training
The training process is similar to the speech generation process described above.
- Full-Parameter Fine-Tuning(Recommended)

```bash
bash sft/train.sh sft/conf/train_asr.yaml
```

- LoRA Fine-Tuning

```bash
bash sft/train.sh sft/conf/train_asr_lora.yaml
```

## Acknowledgements
Our SFT training framework builds upon several excellent open-source projects. We are grateful for their contributions:
- [Accelerate](https://github.com/huggingface/accelerate.git): For its simple and efficient distributed training toolkit.
- [WeNet](https://github.com/wenet-e2e/wenet.git): For the dataset processing implementation which we adapted.
- [Grouped_gemm](https://github.com/fanshiqing/grouped_gemm.git): For providing the high-performance MoE kernels that significantly accelerate our training.
- [HuggingFace Transformers PR #40583](https://github.com/huggingface/transformers/pull/40583): For providing a clear reference on how to integrate grouped_gemm into MoE model.