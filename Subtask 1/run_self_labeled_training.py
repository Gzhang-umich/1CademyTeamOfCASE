from arch1_causality_model import *
from pooler_arch2_causality_model import *
from base_arch2_causality_model import *
from multitask_model import *
from multiset_trainer import *
from multiset_baseline_trainer import *
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    parser = argparse.ArgumentParser(description = "train and evaluate multi-task learning model")
    parser.add_argument("-model_name",
                        help = "Path to pre-trained model base",
                        required = True,
                        type = str)
    parser.add_argument("-arch",
                        help = "Model architecture",
                        required = True,
                        type = str)
    parser.add_argument("-init_train_set",
                        help = ".csv file for training",
                        required = True,
                        type = str)
    parser.add_argument("-train_set",
                        help = ".csv file for training",
                        required = True,
                        type = str)
    parser.add_argument("-eval_set",
                        help = ".csv file for evaluation",
                        required = False,
                        type = str)
    parser.add_argument("-test_set",
                        help = ".csv file for evaluation",
                        required = False,
                        type = str)
    parser.add_argument("-num_epochs",
                        help = "number of epochs for training",
                        required = False,
                        default=3,
                        type = int)
    parser.add_argument("-learning_rate",
                        help = "learning rate for training",
                        required = False,
                        default=5e-5,
                        type = float)
    parser.add_argument("-per_device_train_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for training",
                        required = False,
                        default=8,
                        type = int)
    parser.add_argument("-per_device_eval_batch_size",
                        help = "The batch size per GPU/TPU core/CPU for evaluation",
                        required = False,
                        default=8,
                        type = int)
    parser.add_argument("-results_dir",
                        help = "file to save results",
                        required = False,
                        default='results',
                        type = str)
    args = parser.parse_args()

    if args.arch != 'base':
        entailment_model, event_detection_model, tokenizer = get_multitask_models(args.model_name,
                                    per_device_train_batch_size=args.per_device_train_batch_size)

        if args.arch == '1':
            causality_model = CausalityModel(entailment_model, event_detection_model)
        elif args.arch == '2':
            if args.arch == 'bert-base-cased':
                causality_model = CausalityModel_pooler(entailment_model, event_detection_model)
            else:
                causality_model = CausalityModel_base(entailment_model, event_detection_model)

    elif args.arch == 'base':
        causality_model = AutoModelForSequenceClassification.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    causality_model.to(device)

    init_train_set = tokenize_data(load_csv(args.init_train_set), tokenizer)
    train_set = tokenize_data(load_csv(args.train_set), tokenizer)
    if args.eval_set:
        eval_set = tokenize_data(load_csv(args.eval_set), tokenizer)
    else:
        eval_set=None
    if args.test_set:
        test_set = tokenize_data(load_csv(args.test_set), tokenizer)
    else:
        test_set=None

    if args.arch == 'base':
        trainer = MS_Baseline_Trainer(model=causality_model,
                                    init_train_set=init_train_set,
                                    train_set=train_set,
                                    eval_set=eval_set,
                                    test_set=test_set,
                                    results_dir=args.results_dir,
                                    device=device)

    else:
        trainer = MultiSetTrainer(model=causality_model,
                                init_train_set=init_train_set,
                                train_set=train_set,
                                eval_set=eval_set,
                                test_set=test_set,
                                results_dir=args.results_dir,
                                device=device)

    training_args = {'num_epochs': args.num_epochs,
                    'learning_rate': args.learning_rate,
                    'per_device_train_batch_size': args.per_device_train_batch_size,
                    'per_device_eval_batch_size': args.per_device_eval_batch_size
                    }

    trainer.train_model(**training_args)

if __name__ == '__main__':
    main()
