set -xv
set -eu

unset PythonPath
unset NCCL_DEBUG

export PYTHONPATH="./${PYTHONPATH:+:$PYTHONPATH}"

WORLD_SIZE=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RANDOM_PORT=$[$RANDOM + 20000]
MASTER_PORT=${MASTER_PORT:-${RANDOM_PORT}}

# On Kaggle, you may want to limit/set processes manually based on GPUs if nvidia-smi fails
# Default to 2 for Kaggle T4x2 environments if not detected correctly
NUM_PROCESSES=$(nvidia-smi -L | wc -l)
if [ "$NUM_PROCESSES" -eq 0 ]; then
    echo "Warning: nvidia-smi not found. Defaulting to 1 process."
    NUM_PROCESSES=1
fi

NUM_PROCESSES=$((NUM_PROCESSES*WORLD_SIZE))
#export CUDA_LAUNCH_BLOCKING=1

# Usually accelerate config is defined in the same dir as the yaml config
accelerate_config="$(dirname $1)/accelerate_config.yaml"

train_cmd="accelerate launch \
    --config_file ${accelerate_config} \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes $NUM_PROCESSES \
    --num_machines $WORLD_SIZE \
"

${train_cmd} sft/train_edit_qlora.py $1
