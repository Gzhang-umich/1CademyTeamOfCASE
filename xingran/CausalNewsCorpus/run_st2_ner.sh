GPU_ID=$1
# baseline
CUDA_VISIBLE_DEVICES=$GPU_ID python run_st2.py \
  --learning_rate=1e-05 \
  --model_name_or_path=albert-xxlarge-v2 \
  --num_train_epochs=20 \
  --num_warmup_steps=100 \
  --output_dir=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/checkpoints/test \
  --per_device_train_batch_size=2 \
  --report_to=wandb \
  --task_name=ner \
  --train_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --validation_file=/data/chenxingran/CASE2022/src/1CademyTeamOfCASE/xingran/CausalNewsCorpus/data/dev_subtask2_grouped.csv \
  --weight_decay=0.01
