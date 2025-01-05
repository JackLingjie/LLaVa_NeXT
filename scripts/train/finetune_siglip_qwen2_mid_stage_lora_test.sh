export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="/mnt/lingjiejiang/textual_aesthetics/model_checkpoint/sft_merge_checkpoints/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="Qwen2-7B-Instruct"
VISION_MODEL_VERSION="/mnt/lingjiejiang/multimodal_code/checkpoints/clips/siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="siglip-so400m-patch14-384"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

MAX_MODEL_LEN=11264

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-mid_stage"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-mid_stage_${MAX_MODEL_LEN}_lora"
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint
NUM_GPUS=8

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path scripts/train/mid_stage_mypath.yaml \
    --image_folder /mnt/lingjiejiang/multimodal_code/data/llava_onevision/LLaVA-ReCap-558K \
    --pretrain_mm_mlp_adapter="/mnt/lingjiejiang/multimodal_code/checkpoints/projectors/llavanext-siglip-so400m-patch14-384-Qwen2-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/mnt/lingjiejiang/multimodal_code/checkpoints/vlms/${MID_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MAX_MODEL_LEN} \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    # --attn_implementation sdpa

# You can delete the sdpa attn_implementation if you want to use flash attn
