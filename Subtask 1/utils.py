from datasets import load_dataset

def load_csv(file):
        dataset = load_dataset('csv', data_files=file)
        dataset = dataset['train']
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['text', 'label']])
        return dataset

def tokenize_data(dataset, tokenizer):
    #tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    if 'label' in tokenized_dataset.column_names:
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

def get_encoder_attr_name(model):
    """
    The encoder transformer is named differently in each model "architecture".
    This method lets us get the name of the encoder attribute
    """
    model_class_name = model.__class__.__name__
    if model_class_name.startswith("Bert"):
        return "bert"
    elif model_class_name.startswith("Roberta") or model_class_name.startswith("XLM"):
        return "roberta"
    elif model_class_name.startswith("Electra"):
        return "electra"
    elif model_class_name.startswith("Deberta"):
        return "deberta"
    elif model_class_name.startswith("Albert"):
        return "albert"
    else:
        raise KeyError(f"Add support for new model {model_class_name}")
