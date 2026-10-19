#!/usr/bin/env bash
set -euo pipefail

# Run from the project root
model_variant="best_model.pt"

dataset="${1//-/_}"
task_name="$2"
shift 2

dataset_dir="data/mimic-tokenized"

case "$dataset" in
    mimic*)
        model="layer_6_do_0.3"
        ;;
    *)
        echo "Wrong dataset name: '$dataset', available are: 'mimic'."
        exit 1
        ;;
esac

if [[ ! -d "$dataset_dir" ]]; then
    echo "Dataset directory not found: $dataset_dir"
    exit 1
fi

res_name_prefix=""
rep_num=20
extra_args=()

for arg in "$@"; do
    case "$arg" in
        res_name_prefix=*)
            res_name_prefix="${arg#res_name_prefix=}_"
            ;;
        rep_num=*)
            rep_num="${arg#rep_num=}"
            ;;
        *)
            extra_args+=("$arg")
            ;;
    esac
done

# Expose two GPUs to the process
export CUDA_VISIBLE_DEVICES=5d
export NUM_GPUS=1

echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"

output_dir="results/${task_name}/${dataset}_${model}_${model_variant%_*}"
output_fn="${res_name_prefix}rep_size_${rep_num}_$(date +%Y-%m-%d_%H-%M-%S)"

run_inference() {
    ethos_infer \
        "task=${task_name}" \
        "model_fp=${dataset_dir}/models/${model}/${model_variant}" \
        "input_dir=${dataset_dir}/test" \
        "output_dir=${output_dir}" \
        "output_fn=${output_fn}" \
        "${extra_args[@]}" \
        n_gpus=1 \
        "rep_num=${rep_num}"
}

if command -v singularity >/dev/null 2>&1; then
    singularity exec \
        --nv \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -lc '
            set -e
            export PATH="$HOME/.local/bin:$PATH"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/:${LD_LIBRARY_PATH:-}"
            export TORCHINDUCTOR_CACHE_DIR=/ethos/torchinductor_cache
            cd /ethos

            # Install the project if it is not already installed in the container
            pip install --no-deps --no-index --no-build-isolation --user -e . >/dev/null

            exec ethos_infer "$@"
        ' bash \
        "task=${task_name}" \
        "model_fp=/ethos/${dataset_dir}/models/${model}/${model_variant}" \
        "input_dir=/ethos/${dataset_dir}/test" \
        "output_dir=/ethos/${output_dir}" \
        "output_fn=${output_fn}" \
        "${extra_args[@]}" \
        n_gpus=1 \
        "rep_num=${rep_num}"
else
    run_inference
fi