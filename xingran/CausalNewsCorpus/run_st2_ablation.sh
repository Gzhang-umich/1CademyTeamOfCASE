# ablation study for top-k
# baseline + BSS + ES + DA

GPU_ID=$1

for topk in 1 2 3 5 10
do
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
    --pretrained_signal_detector \
    --topk $topk
done