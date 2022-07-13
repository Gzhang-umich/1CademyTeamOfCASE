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

python run_ner_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --train_file /data/chenxingran/CASE2022/src/CausalNewsCorpus/data/train_subtask2_grouped.csv \
  --text_column_name text \
  --label_column_name causal_text_w_pairs \
  --task_name ner \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /tmp/$TASK_NAME/