The two architectures for multi-task learning are similar in that they both use
two task specific heads, pre-trained on a event detection dataset and am
entailment dataset, respectively. They differ in that architecture #1 simply
combines the output of the two heads and passes those logits into a linear
layer, whereas architecture #2 has its own casualty classification head
(modeled after the sequence classification heads from huggingface) and the
outputs if the casualty classification head and the event and entailment
detection heads are combined before going through a linear layer.

To train a model on architecture #1, use the script:

python train_multitask_arch1.py \
  -entailment_detection_model=adamnik/bert-entailment-detection \
  -event_detection_model=adamnik/bert-event-detection \
  -tokenizer=bert-base-cased \
  -train_set=<train_set.csv> \
  -test_set=<dev_set.csv> \
  -num_epochs=3 \
  -results_file=<results_dir>

  or with the models:
    adamnik/electra-entailment-detection
    adamnik/electra-event-detection
    tokenizer: google/electra-base-discriminator

For architecture #2 use the model specific file scripts:

python bert_train_multitask_arch2.py \
  -train_set=<train_set.csv> \
  -test_set=<dev_set.csv> \
  -num_epochs=3 \
  -results_file=<results_dir>

python electra_train_multitask_arch2.py \
  -train_set=<train_set.csv> \
  -test_set=<dev_set.csv> \
  -num_epochs=3 \
  -results_file=<results_dir>
