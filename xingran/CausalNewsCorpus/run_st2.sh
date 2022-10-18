# final ablation experiments
GPU_ID=$1
TRAIN_FILE=$2  # should be .../1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv
VALIDATION_FILE=$3  # should be .../1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv 
OUTPUT_DIR=$4
AUGMENTATION_FILE=$5  # should be .../1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv or 
                      # .../augmented_subtask2_9.csv for two different settings of data augmentation.


# baseline
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir= \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --weight_decay=0.01

# # baseline + BSS
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
  --weight_decay=0.01 \
  --postprocessing_position_selector \
  --beam_search


# # baseline + JS
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
  --weight_decay=0.01 \
  --signal_classification


# # baseline + ES
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
  --weight_decay=0.01 \
  --postprocessing_position_selector \
  --pretrained_signal_detector


# # baseline + BSS + ES + DA
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --augmentation_file  \
  --weight_decay=0.01 \
  --postprocessing_position_selector \
  --beam_search \
  --signal_classification \
  --pretrained_signal_detector

# # baseline + DA
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --augmentation_file $AUGMENTATION_FILE \
  --weight_decay=0.01

# # baseline + DA + BSS
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --augmentation_file $AUGMENTATION_FILE \
  --weight_decay=0.01 \
  --postprocessing_position_selector \
  --beam_search

# # baseline + DA + ES
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --augmentation_file $AUGMENTATION_FILE \
  --weight_decay=0.01 \
  --signal_classification \
  --pretrained_signal_detector

# # baseline + DA + JS
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR \
  --per_device_train_batch_size=100 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=$TRAIN_FILE \
  --validation_file=$VALIDATION_FILE \
  --augmentation_file $AUGMENTATION_FILE \
  --weight_decay=0.01 \
  --signal_classification
