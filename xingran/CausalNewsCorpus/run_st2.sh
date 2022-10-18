### Cause-Effect Span Detection

## We used the UniCausal (https://github.com/tanfiona/UniCausal) repository to run the baselines
## The repository is currently in private mode due to anonymity requirements for submission
# Train and Test CNC using Individual Token Baseline Model
# cnc_train.csv // cnc_test.csv files must exist in data/splits folder
# sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run_tokbase.py \
# --dataset_name cnc --model_name_or_path bert-base-cased \
# --output_dir outs/cnc/dev --label_column_name ce_tags \
# --num_train_epochs 20 --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 32 --do_train_val --do_predict --do_train

### Signal Prediction
# No baseline model implemented! Just take random, using `random_st2.py` for now

# CUDA_VISIBLE_DEVICES=$1 python run_st2_v2.py \
#   --model_name_or_path albert-xxlarge-v2 \
#   --train_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
#   --validation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
#   --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_9.csv \
#   --task_name ner \
#   --dropout 0.2 \
#   --seed 42 \
#   --max_length 256 \
#   --per_device_train_batch_size 16 \
#   --learning_rate 1e-5 \
#   --num_train_epochs 30 \
#   --num_warmup_steps 100 \
#   --weight_decay 0.01 \
#   --output_dir /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/baseline+postprocess+pretrained-signal-detector-augment-9-v2.0 \
#   --report_to wandb \
#   --signal_classification \
#   --pretrained_signal_detector \
#   --postprocessing_position_selector \
#   --beam_search
  
#   # --do_test \
#   # --load_checkpoint_for_test /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/baseline+postprocess+pretrained-signal-detector-augment-9/epoch_17/pytorch_model.bin
  
#   # --validation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/test_subtask2_text.csv \

# final ablation experiments
GPU_ID=$1
# baseline
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --weight_decay=0.01

# # baseline + BSS
# CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
#   --dropout=0.2 \
#   --learning_rate=1e-05 \
#   --model_name_or_path=albert-xxlarge-v2 \
#   --num_train_epochs=20 \
#   --num_warmup_steps=0 \
#   --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
#   --per_device_train_batch_size=4 \
#   --report_to=wandb \
#   --task_name=ner \
#   --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
#   --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
#   --weight_decay=0.01 \
#   --postprocessing_position_selector \
#   --beam_search


# # baseline + JS
# CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
#   --dropout=0.2 \
#   --learning_rate=1e-05 \
#   --model_name_or_path=albert-xxlarge-v2 \
#   --num_train_epochs=20 \
#   --num_warmup_steps=0 \
#   --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
#   --per_device_train_batch_size=4 \
#   --report_to=wandb \
#   --task_name=ner \
#   --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
#   --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
#   --weight_decay=0.01 \
#   --signal_classification


# # baseline + ES
# CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
#   --dropout=0.2 \
#   --learning_rate=1e-05 \
#   --model_name_or_path=albert-xxlarge-v2 \
#   --num_train_epochs=20 \
#   --num_warmup_steps=0 \
#   --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
#   --per_device_train_batch_size=4 \
#   --report_to=wandb \
#   --task_name=ner \
#   --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
#   --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
#   --weight_decay=0.01 \
#   --postprocessing_position_selector \
#   --pretrained_signal_detector


# # baseline + BSS + ES + DA
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv \
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
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv \
  --weight_decay=0.01

# # baseline + DA + BSS
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2_v2.py \
  --dropout=0.2 \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv \
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
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=16 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv \
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
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=100 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --augmentation_file /data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/augmented_subtask2_4.csv \
  --weight_decay=0.01 \
  --signal_classification
