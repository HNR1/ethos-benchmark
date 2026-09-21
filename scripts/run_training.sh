#!/bin/bash -l
#SBATCH --job-name=ethos_train::proj=IRB2023P002279,
#SBATCH --time=3-0
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --output=ethos_train.log

# this script is intended to be run from the project root with: GPU_IDS=1,2 uv run -- bash scripts/run_training.sh device=cuda
export OMP_NUM_THREADS=1

dataset="mimic_ed"
dataset_name="mimic"

data_path=../../../mnt/data_share/project_henri/ethos-ares/mimic-tokenized
clear
if [[ ! -d $data_path ]]; then
    echo "Dataset directory not found: $data_path"
    exit 1
fi

GPU_IDS="${GPU_IDS:-}"

# shift 1

BATCH_SIZE=32
N_POSITIONS=2048
N_LAYER=6
N_HEAD=12
N_EMBD=768
DROPOUT=0.3
LR=0.0006
MIN_LR=0.00001

model_name="layer_${N_LAYER}_do_${DROPOUT}"

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null

# Use other tmp dir to avoid /tmp filling up and preserve the cache across the runs
export TORCHINDUCTOR_CACHE_DIR=/ethos/torchinductor_cache
"

script_body="
torchrun --no_python --standalone --nproc_per_node=\${NUM_GPUS} ethos_train \
  data_fp=$data_path/train \
  data_fp_val=$data_path/val \
  val_size=6 \
  batch_size=$BATCH_SIZE \
  n_positions=$N_POSITIONS \
  n_layer=$N_LAYER \
  n_head=$N_HEAD \
  n_embd=$N_EMBD \
  dropout=$DROPOUT \
  lr=$LR \
  min_lr=$MIN_LR \
  log_interval=10 \
  eval_interval=1500 \
  gradient_accumulation_steps=12 \
  warmup_iters=5000 \
  max_iters=200000 \
  lr_decay_iters=100000 \
  wandb_log=true \
  wandb_project="ethos-meds-$dataset_name" \
  wandb_run_name=$model_name \
  $* \
  out_dir="${data_path}/models/${model_name}"
"

module load singularity 2>/dev/null

# GPU selection
# Leave empty to use all GPUs available to the job.
if [[ -n "${GPU_IDS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

# Count GPUs visible to the process.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    NUM_GPUS=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
elif [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE}"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
fi

export NUM_GPUS

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
echo "NUM_GPUS=${NUM_GPUS}"

if command -v singularity >/dev/null; then

    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"

else

    bash -c "${script_body}"

fi
