#!/usr/bin/env python
# coding: utf-8
# The code for this baseline was based on the following Huggingface tutorial:
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb


################################ Imports #####################################
import os
import json
import torch
import numpy as np

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader


############################### Utility functions ###############################

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], padding = True, truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["label_ids"]):
        #word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        #word_idx = None
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


#################### Environment variables and training hyperparameters ################

cuda_device = "0"
dataset = "econ" # full, cs, phys, econ
MODEL_TYPE = "bert-base-uncased"
label_all_tokens = False # Set to True to label all wordpieces
batch_size = 16
learning_rate = 2e-5
num_epochs = 5
weight_decay = 0.01

# Setting the cude device
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} as device")

# empty cache
torch.cuda.empty_cache()


######################## Loading the dataset #########################################

data_files = {
        "train": f"full_jargon_dataset/updated_{dataset}_train.json", 
        "val": f"full_jargon_dataset/updated_{dataset}_dev.json"}
datasets = load_dataset("json", data_files=data_files)

label_list = ['O', 'B-Jargon']
label2id = {"O": 0, "B-Jargon": 1}
id2label = {id: tag for tag, id in label2id.items()}


########################### Tokenizer #################################################
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)

# Tokenizing datasets
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched = True, remove_columns=datasets["train"].column_names)
# This data collator will do variable padding
data_collator = DataCollatorForTokenClassification(tokenizer)

##########################  Model & Training ######################################################3
model = AutoModelForTokenClassification.from_pretrained(MODEL_TYPE, num_labels=len(label2id))
model_name = MODEL_TYPE.split("/")[-1]

# Creating raining arguments for Trainer
args = TrainingArguments(
    f"{model_name}-finetuned-jargon",
    evaluation_strategy = "epoch",
    learning_rate = learning_rate,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    num_train_epochs = num_epochs,
    weight_decay = weight_decay,
    push_to_hub = False,
)

#Creating Trainer object
trainer = Trainer(
    model,
    args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["val"],
    data_collator = data_collator,
    tokenizer = tokenizer,
    compute_metrics = compute_metrics
)


# Training
trainer.train()

# Evaluation
trainer.evaluate()


############################### Metric ######################################
metric = load_metric("seqeval")

# Obtain final predictions
predictions, labels, _ = trainer.predict(tokenized_datasets["val"])
predictions = np.argmax(predictions, axis=2)

# Remove ignored index (special tokens)
true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
                ]
true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
                ]

results = metric.compute(predictions=true_predictions, references=true_labels)
print(results)
