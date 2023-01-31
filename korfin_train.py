'''
deepspeed --num_gpus=2 korfin_train.py

'''
import os
import pandas as pd
import numpy as np
import random
import torch
from datasets import load_dataset
from argparse import ArgumentParser
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from deepspeed.comm import comm
import deepspeed
import wandb

random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
comm.init_distributed("nccl")
torch.cuda.set_device(torch.distributed.get_rank())

parser = ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--model_name", type=str, default="lighthouse/mdeberta-v3-base-kor-further")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", default=20, type=int)
parser.add_argument("--batch_size", default=64, type=int)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=3,
).cuda()

dataset = load_dataset("amphora/korfin-asc")

title = dataset["train"]["SRC"] 

sent_list = []
for sen in dataset["train"]["SENTIMENT"]:
    sent = sent_list.append(sen.replace("NEGATIVE", "0").replace("NEUTRAL", "1").replace("POSITIVE", "2"))

dataset = [
    {"data": str(t), "label": int(s)} 
    for t, s in zip(title, sent_list)
]

train_data = dataset[:int(len(dataset)*0.8)]
eval_data = dataset[int(len(dataset)*0.8):]

train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    num_workers=os.cpu_count() // dist.get_world_size(),
    drop_last=True,
    pin_memory=False,
    shuffle=False,
    sampler=DistributedSampler(
        train_data,
        shuffle=True,
        drop_last=True,
        seed=random_seed,
    ),
)

eval_loader = DataLoader(
    eval_data,
    batch_size=args.batch_size,
    num_workers=os.cpu_count() // dist.get_world_size(),
    drop_last=True,
    pin_memory=False,
    shuffle=False,
    sampler=DistributedSampler( 
        eval_data,
        shuffle=True,
        drop_last=True,
        seed=random_seed,
    ),
)

no_decay = [
    "bias",
    "LayerNorm.weight",
]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 3e-7,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=optimizer_grouped_parameters,
)

if dist.get_rank() == 0:
    wandb.init(project="KPMG", name=f"{args.model_name}_korfin")
    
for epoch in range(args.epoch):
    for train in tqdm(train_loader):
        model.train()
        engine.zero_grad()
        text, label = train["data"], train["label"].cuda()
        tokens = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        output = engine.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )

        loss = output.loss
        engine.backward(loss)
        engine.step()
        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1
                
    if dist.get_rank() == 0:
        wandb.log({"loss": loss})
        wandb.log({"epoch": epoch})
        wandb.log({"acc": acc / len(classification_results)})

    
    with torch.no_grad():
        model.eval()
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"].cuda()
            eval_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            eval_input_ids = eval_tokens.input_ids.cuda()
            eval_attention_mask = eval_tokens.attention_mask.cuda()

            eval_out = engine.forward(
                input_ids=eval_input_ids,
                attention_mask=eval_attention_mask,
                labels=eval_label
            )
            eval_loss = eval_out.loss
            eval_classification_results = eval_out.logits.argmax(-1)
            
            eval_acc = 0
            for res, lab in zip(eval_classification_results, eval_label):
                if res == lab:
                    eval_acc += 1
                    
        if dist.get_rank() == 0: 
            wandb.log({"eval_loss": eval_loss})
            wandb.log({"eval_acc": eval_acc / len(eval_classification_results)})

        
        ckpt_dir = f"model_save/{args.model_name.replace('/', '-')}-{epoch}-korfin"
        model.save_pretrained(ckpt_dir)


        # torch.save(
        #     model.state_dict(),
        #     f"model_save/{model_name.replace('/', '-')}-{epoch}-test-large.pt",
        # )
