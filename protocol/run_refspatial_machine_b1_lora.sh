#!/usr/bin/env bash
set -euo pipefail

workspace="/home/dhcn/ur_ws/src/myproject"
repo="${workspace}/RoboRefer"
python_bin="${workspace}/.conda-roborefer/bin/python"
mode="${1:-smoke}"

case "${mode}" in
  smoke)
    output_dir="${workspace}/results/spatial_vlm_refspatial_v1/wp4_b1_lora_smoke/model"
    max_steps=1
    epochs=1
    grad_accum=1
    save_strategy="no"
    ;;
  pilot)
    output_dir="${workspace}/results/spatial_vlm_refspatial_v1/wp4_b1_lora_pilot/model"
    max_steps=-1
    epochs=1
    grad_accum=8
    save_strategy="epoch"
    ;;
  *)
    echo "Usage: $0 [smoke|pilot]" >&2
    exit 2
    ;;
esac

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
  --data_mixture refspatial_machine_b1_pilot_rgbd_v1 \
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
  --num_train_epochs "${epochs}" \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${grad_accum}" \
  --eval_strategy no \
  --save_strategy "${save_strategy}" \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --report_to none \
  --remove_unused_columns False
