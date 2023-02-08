import os
import sys
import random
import torch
import json
from argparse import ArgumentParser

from ner_dataloader import DataloaderForNER
from ner_metric import Seqeval
from utils import seed_everything

from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

import wandb

def define_args():
    p = ArgumentParser()

    p.add_argument('--model_name', type=str, default="lighthouse/mdeberta-v3-base-kor-further")
    p.add_argument('--n_epoch', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--max_length', type=int, default=120)
    p.add_argument('--label_type', type=str, default=None)
    p.add_argument('--config', type=str, required=True)
    training_config = json.load(open(p.parse_args().config))
    config = p.parse_args()
    return config, training_config

def train(config, training_config):
    dataloader = DataloaderForNER(config.model_name, max_length=config.max_length, label_type=config.label_type)
    model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=len(dataloader.label_to_id))
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_datasets, eval_datasets = dataloader.load_data()
    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = Seqeval(list(dataloader.label_to_id.keys()))

    wandb.init(project="kpmg_ner", name=f"{config.model_name}_{config.label_type}")
    output_dir = f"{config.model_name.replace('/', '-')}_{config.label_type}"
    training_args = TrainingArguments(
        output_dir,
        **training_config["experiment"]
    )

    wandb.config.update(training_args)
    
    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_datasets,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric.compute_seqeval
    )

    trainer.train()
    trainer.evaluate()
    wandb.finish()

config, training_config = define_args()
seed_everything(config.random_seed)
train(config, training_config)