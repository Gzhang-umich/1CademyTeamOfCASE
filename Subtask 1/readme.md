To run a baseline model, run run_baseline.py.

python run_baseline.py \
  -model_name= <name of huggingface model> \
  -train_set=<.csv file of training set> \
  -eval_set=<.csv file of evaluation set> \
  -test_set=<.csv file of evaluation set> \
  -num_epochs=5 \
  -learning_rate \
  -per_device_train_batch_size=8 \
  -per_device_eval_batch_size=8 \
  -results_dir=<results_dir> \

To run a model using the self-training paradigm with both self-labeled examples and the original training data, run run_self_labeled_training.py. Supports both base pre-trained models and multi-task learning architectures.

python run_self_labeled_training.py \
  -model_name= <name of huggingface model> \
  -arch=<1, 2, or base>
  -init_train_set=<.csv file of self labeled examples training set> \
  -train_set=<.csv file of training set> \
  -eval_set=<.csv file of evaluation set> \
  -test_set=<.csv file of evaluation set> \
  -num_epochs=5 \
  -learning_rate \
  -per_device_train_batch_size=8 \
  -per_device_eval_batch_size=8 \
  -results_dir=<results_dir> \

To run one of our multi-task learning architectures on only one train set, run run_mtl.py.

python run_mtl.py \
  -model_name= <name of huggingface model> \
  -arch=<1 or 2>
  -train_set=<.csv file of training set> \
  -eval_set=<.csv file of evaluation set> \
  -test_set=<.csv file of evaluation set> \
  -num_epochs=5 \
  -learning_rate \
  -per_device_train_batch_size=8 \
  -per_device_eval_batch_size=8 \
  -results_dir=<results_dir> \
