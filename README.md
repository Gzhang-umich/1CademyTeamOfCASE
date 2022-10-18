# 1CademyTeamOfCASE for Subtask 2

This is the official implementation of the 1CademyTeamOfCASE for the [Causal News Corpus - Event Causality Shared Task 2022](https://codalab.lisn.upsaclay.fr/competitions/2299#results).


## Train for Subtask 2

You can use commands in ```run_st2.sh``` to train the model under different settings. Note that we use the wandb to log the training result. If you do not want to use it, feel free to comment all the lines related to wandb.


## Signal Detector for ES

We have upload the model checkpoint to the Huggingface Hub, and our code can automatically download and use it once the ES setting is chosen. You can also download it via the following code:
```
model = AutoModelForSequenceClassification.from_pretrained('chenxran/case2022-signal-detector')
```