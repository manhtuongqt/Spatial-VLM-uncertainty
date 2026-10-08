#!/usr/bin/env bash
set -euo pipefail

workspace="/home/dhcn/ur_ws/src/myproject"
repo="${workspace}/RoboRefer"
python_bin="${workspace}/.conda-roborefer/bin/python"
release="${workspace}/datasets/D_tabletop_clean_v1"
run_root="${workspace}/results/spatial_vlm_refspatial_v1/wp7_b1_clean_grounding"
mode="${1:-smoke}"

"${python_bin}" -s "${workspace}/protocol/refspatial_wp1_clean_release.py" --check

case "${mode}" in
  smoke)
    output_dir="${run_root}/smoke_model"
    max_steps=1
    grad_accum=1
    save_strategy="no"
    ;;
  full)
    output_dir="${run_root}/model"
    max_steps=-1
    grad_accum=4
    save_strategy="steps"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${release}/manifest.json" ]]; then
  echo "Missing clean release manifest" >&2
  exit 1
fi

mkdir -p "${output_dir}"
export LD_LIBRARY_PATH="${workspace}/.conda-roborefer/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

cd "${repo}"
exec "${python_bin}" -s llava/train/train.py \
  --model_name_or_path "${repo}/models/RoboRefer-2B-SFT" \
  --chat_template qwen2 \
  --version qwen2 \
  --data_mixture refspatial_clean_b1_grounding_rgbd_v1 \
  --enable_depth True \
  --use_depth_tower True \
  --mm_vision_select_feature cls_patch \
  --mm_vision_select_layer -2 \
  --image_aspect_ratio dynamic \
  --bits 4 \
  --double_quant True \
  --quant_type nf4 \
  --bf16 True \
  --tf32 True \
  --lora_enable True \
  --lora_llm True \
  --lora_vt False \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --tune_language_model False \
  --tune_vision_tower False \
  --tune_mm_projector False \
  --tune_depth_tower False \
  --tune_depth_projector False \
  --output_dir "${output_dir}" \
  --num_train_epochs 1 \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${grad_accum}" \
  --eval_strategy no \
  --save_strategy "${save_strategy}" \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --seed 9092026 \
  --data_seed 9092026 \
  --report_to none \
  --remove_unused_columns False
