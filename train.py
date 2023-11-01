import argparse
import json
import os
import os.path as op

import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding,
                          Trainer, TrainingArguments
                         )


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--val_split_size", type=float, default=0.9)

args = parser.parse_args()

assert op.exists(args.dataset), "Dataset file doesn't exist"
assert op.exists(args.output_folder), "Output folder file doesn't exist"

tokenizer = AutoTokenizer.from_pretrained(args.model_name)


def preprocess(examples):
    global tokenizer
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="micro")


def main(args):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # with open(args.dataset, 'r') as f:
    #     dataset = json.load(f)

    dataset = load_dataset("json", data_files=args.dataset)["train"]

    dataset = dataset.shuffle(seed=42)
    dataset = dataset.train_test_split(test_size=0.1)

    train = dataset["train"]
    val = dataset["test"]

    train = train.map(preprocess, batched=True)
    val = val.map(preprocess, batched=True)

    id2label = {0: "world", 1: "sport", 2: "business", 3: "sci-tech"}
    label2id = {"world": 0, "sport": 1, "business": 2, "sci-tech": 3}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=4, id2label=id2label, label2id=label2id
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    training_args = TrainingArguments(
        output_dir=args.output_folder,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print(train)

    trainer.train()


if __name__ == '__main__':
    main(args)
