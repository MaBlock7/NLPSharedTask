import os
import evaluate
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    pipeline,
)
from datasets import Dataset

def finetune_and_save(datapath: str,  save_name: str, save_dir: str = '.', base_model: str = 'allenai/scibert_scivocab_uncased'):
    base_dir = save_dir
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device} to finetune {base_model} on dataset at {datapath}")

    label2id={n: n+1 for n in range(0,17)}
    id2label={n: n-1 for n in range(1,18)}

    model_path = base_model
    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=18,
        return_dict=True)

    # 'allenai/scibert_scivocab_uncased'
    # 'FacebookAI/xlm-roberta-large'
    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # LOAD DATA
    df = pd.read_csv(datapath)
    df = df.dropna(subset='text_clean')

    # CREATE TRAIN/TEST SPLIT
    def tokenize_text(texts):
        return tokenizer(texts, truncation=True, max_length=256, padding="max_length", return_tensors="pt")

    def rule_based_train_test_split(
        data: pd.DataFrame,
        label_col: str = 'label',
        test_size: float = 0.3,
        random_state: int | None = None
    ) -> dict:
        """Creates train-test split that makes sure that at least two abstracts for each id are in the test set."""

        abstract_data = data[data.is_abstract == 1]

        # Randomly sample 2 abstracts per sdg group
        test_a = abstract_data.groupby(label_col).sample(n=1, random_state=random_state)

        # Remove the entries already in the test set from the rest of the data
        data = data[~data.index.isin(test_a.index)].copy()

        # Split the remaining data into train and test
        train, test_b = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[label_col])

        # Concatenate both test sets and shuffle them again
        test = pd.concat([test_a, test_b]).sample(frac=1).reset_index(drop=True)

        return train, test


    def split_data(
        data: pd.DataFrame
    ):
        train = data[~(data.is_abstract == 1)]
        test = data[data.is_abstract == 1]

        return train, test

    # Apply tokenizer
    tokenized_output = tokenize_text(df['text_clean'].to_list())
    """
    df_tokenized = pd.DataFrame({
        'raw_text': df['text_clean'].tolist(),
        'input_ids': list(tokenized_output['input_ids']),
        'attention_mask': list(tokenized_output['attention_mask']),
        'token_type_ids': list(tokenized_output.get('token_type_ids', [[]]*len(df))),
        'label': df['label'].tolist(),
        'is_abstract': df['is_abstract'].to_list()
    })
    """

    df_tokenized = pd.DataFrame({
        'raw_text': df['text_clean'].tolist(),
        'input_ids': [i.tolist() for i in tokenized_output['input_ids']],
        'attention_mask': [i.tolist() for i in tokenized_output['attention_mask']],
        'token_type_ids': [i.tolist() for i in tokenized_output.get('token_type_ids', torch.zeros((len(df), 256), dtype=torch.long))],
        'label': df['label'].tolist(),
        'is_abstract': df['is_abstract'].to_list()
    })

    train_df, test_df = split_data(df_tokenized)

    train_df[['raw_text', 'label', 'is_abstract']].to_csv(os.path.join(base_dir, f"train_df_{model_path.split('/')[-1]}.csv"))
    test_df[['raw_text', 'label', 'is_abstract']].to_csv(os.path.join(base_dir, f"test_df_{model_path.split('/')[-1]}.csv"))

    train_dataset = Dataset.from_pandas(train_df[['input_ids', 'attention_mask', 'token_type_ids', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['input_ids', 'attention_mask', 'token_type_ids', 'label']])
    # FINE-TUNING
    # For training, use the suggested values from the paper:
    #
    # In all settings, we apply a dropout of 0.1 and optimize cross entropy loss using Adam (Kingma and Ba, 2015). We finetune for 2 to 5 epochs using a batch size of 32 and a learning rate of 5e-6, 1e-5, 2e-5, or 5e-5 with a slanted triangular schedule (Howard and Ruder, 2018) which is equivalent to the linear warmup followed by linear decay (Devlin et al., 2019). For each dataset and BERT variant, we pick the best learning rate and number of epochs on the development set and report the corresponding test results. We found the setting that works best across most datasets and models is 2 or 4 epochs and a learning rate of 2e-5. While task-dependent, optimal hyperparameters for each task are often the same across BERT variants.
    # Multiple class prediction (one prediction)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = evaluate.load("accuracy")
        return {
            "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        }
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir, 'models/results'),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,  # As best setting suggested 2 or 4
        warmup_steps=500,  # Slanted triangular schedule start
        learning_rate=2e-5,  # Best learning rate as suggested in the paper
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        lr_scheduler_type='linear',  # Corresponds to linear warmup followed by linear decay
        load_best_model_at_end=True
    )
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Adam Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Multiple class Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )
    print("Starting training...")
    # Start training
    torch.cuda.empty_cache()
    trainer.train()
    # Saving the model
    output_path = os.path.join(save_dir, save_name)
    trainer.save_model(output_path)
    print(f"Saved model to {output_path}")

    # Saving the tokenizer associated with the model
    tokenizer.save_pretrained(model_path)


if __name__ == '__main__':
    base_models = {
        'muppet': 'facebook/muppet-roberta-large',
        'scibert': 'allenai/scibert_scivocab_uncased'
    }
    chosen = 'muppet'
    base_model = base_models[chosen]

    dataset_names = [
        'gpt-4-turbo',
        'llama-3-70b',
        'mixtral-8x22B',
        'ensemble'
    ]
    index = 3
    name = dataset_names[index]
    dataset = f'data_with_null_with_synth_{name}.csv'
    finetuned_name = f'{chosen}_{name}'
    print(f"Finetuning {chosen} on {name}")

    finetune_and_save(dataset, finetuned_name, base_model=base_model)