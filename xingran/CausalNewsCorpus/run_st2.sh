# final ablation experiments
GPU_ID=$1
TASK=$2
TRAIN_FILE=data/train_subtask2_grouped.csv
VALIDATION_FILE=data/dev_subtask2_grouped.csv 
OUTPUT_DIR=checkpoints/$TASK
AUGMENTATION_FILE=data/augment_subtask2_4.csv


# baseline
if [[ "$TASK" == "baseline" ]]; then
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
    --dropout=0.3 \
    --learning_rate=2e-05 \
    --model_name_or_path=albert-xxlarge-v2 \
    --num_train_epochs=20 \
    --num_warmup_steps=200 \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=1 \
    --report_to=wandb \
    --task_name=ner \
    --train_file=$TRAIN_FILE \
    --validation_file=$VALIDATION_FILE \
    --weight_decay=0.005
elif [[ "$TASK" == "DA+BSS" ]]; then
  # baseline + DA + BSS
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
    --dropout=0.2 \
    --learning_rate=1e-05 \
    --model_name_or_path=albert-xxlarge-v2 \
    --num_train_epochs=20 \
    --num_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=4 \
    --report_to=wandb \
    --task_name=ner \
    --train_file=$TRAIN_FILE \
    --validation_file=$VALIDATION_FILE \
    --augmentation_file $AUGMENTATION_FILE \
    --weight_decay=0.01 \
    --postprocessing_position_selector \
    --beam_search
elif [[ "$TASK" == "DA+ES" ]]; then
  # baseline + DA + ES
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
    --dropout=0.2 \
    --learning_rate=1e-05 \
    --model_name_or_path=albert-xxlarge-v2 \
    --num_train_epochs=20 \
    --num_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=4 \
    --report_to=wandb \
    --task_name=ner \
    --train_file=$TRAIN_FILE \
    --validation_file=$VALIDATION_FILE \
    --augmentation_file $AUGMENTATION_FILE \
    --weight_decay=0.01 \
    --signal_classification \
    --pretrained_signal_detector
elif [[ "$TASK" == "DA+JS" ]]; then
  # baseline + DA + JS
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
    --dropout=0.2 \
    --learning_rate=1e-05 \
    --model_name_or_path=albert-xxlarge-v2 \
    --num_train_epochs=20 \
    --num_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=4 \
    --report_to=wandb \
    --task_name=ner \
    --train_file=$TRAIN_FILE \
    --validation_file=$VALIDATION_FILE \
    --augmentation_file $AUGMENTATION_FILE \
    --weight_decay=0.01 \
    --signal_classification
elif [[ "$TASK" == "final" ]]; then
  # baseline + BSS + ES + DA
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
    --dropout=0.2 \
    --learning_rate=1e-05 \
    --model_name_or_path=albert-xxlarge-v2 \
    --num_train_epochs=20 \
    --num_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --per_device_train_batch_size=4 \
    --report_to=wandb \
    --task_name=ner \
    --train_file=$TRAIN_FILE \
    --validation_file=$VALIDATION_FILE \
    --augmentation_file=$AUGMENTATION_FILE \
    --weight_decay=0.01 \
    --postprocessing_position_selector \
    --beam_search \
    --signal_classification \
    --pretrained_signal_detector
fi