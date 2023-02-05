import os
import random
import torch
import json
from argparse import ArgumentParser
from NER_dataloader import Loader, seed_everything
from Metric_ner import Metric
from datasets import load_metric
from Preprocessor_ner import Preprocessor
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import wandb
import deepspeed
from deepspeed.comm import comm

os.environ["TOKENIZERS_PARALLELISM"] = "True"

parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="lighthouse/mdeberta-v3-base-kor-further")
parser.add_argument("--epoch", default=50, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--randomseed", default=32, type=int)
parser.add_argument("--max_length", default=120, type=int)
parser.add_argument("--config", "-c", type=str, required=True)
config = json.load(open(parser.parse_args().config))
args = parser.parse_args()

seed_everything(args.randomseed)

model = AutoModelForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=2,
).cuda()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)

loader = Loader(args.max_length, args.model_name)
train_datasets = loader.train_load()
eval_datasets = loader.eval_load()

preprocessor = Preprocessor()
train_datasets = train_datasets.map(preprocessor, batched=True)
eval_datasets = eval_datasets.map(preprocessor, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

wandb.init(project="KPMG", name=f"{args.model_name}_NER")
run_name = f"{args.model_name.replace('/', '-')}_NER"

training_args = TrainingArguments(
    run_name,
    num_train_epochs=config["experiment"]["num_epochs"],
    per_device_train_batch_size=config["experiment"]["train_batch_size"],
    per_device_eval_batch_size=config["experiment"]["valid_batch_size"],
    gradient_accumulation_steps=config["experiment"]["gradient_accumulation_steps"],
    learning_rate=config["experiment"]["learning_rate"],
    weight_decay=config["experiment"]["weight_decay"],
    warmup_ratio=config["experiment"]["warmup_ratio"],
    fp16=config["experiment"]["fp16"],
    evaluation_strategy=config["experiment"]["evaluation_strategy"],
    save_steps=config["experiment"]["save_steps"],
    eval_steps=config["experiment"]["eval_steps"],
    logging_steps=config["experiment"]["logging_steps"],
    save_strategy=config["experiment"]["save_strategy"],
    save_total_limit=config["experiment"]["num_checkpoints"],
    load_best_model_at_end=config["experiment"]["load_best_model_at_end"],
    metric_for_best_model=config["experiment"]["metric_for_best_model"],
)

wandb.config.update(training_args)

# Metrics
metrics = Metric()

# Trainer
trainer = Trainer(
    model,
    training_args,
    train_dataset=train_datasets,
    eval_dataset=eval_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=metrics.compute_metrics,
)
# Training
trainer.train()
# Evaluating
trainer.evaluate()
wandb.finish()
















# train_loader = DataLoader(
#     train_datasets,
#     batch_size=args.batch_size,
#     num_workers=os.cpu_count() // dist.get_world_size(),
#     drop_last=True,
#     pin_memory=False,
#     shuffle=False,
#     sampler=DistributedSampler(
#         train_datasets,
#         shuffle=True,
#         drop_last=True,
#         seed=args.randomseed,
#     ),
# )

# eval_loader = DataLoader(
#     eval_datasets,
#     batch_size=args.batch_size,
#     num_workers=os.cpu_count() // dist.get_world_size(),
#     drop_last=True,
#     pin_memory=False,
#     shuffle=False,
#     sampler=DistributedSampler( 
#         eval_datasets,
#         shuffle=True,
#         drop_last=True,
#         seed=args.randomseed,
#     ),
# )

# no_decay = [
#     "bias",
#     "LayerNorm.weight",
# ]
# optimizer_grouped_parameters = [
#     {
#         "params": [
#             p
#             for n, p in model.named_parameters()
#             if not any(nd in n for nd in no_decay)
#         ],
#         "weight_decay": 3e-7,
#     },
#     {
#         "params": [
#             p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
#         ],
#         "weight_decay": 0.0,
#     },
# ]

# engine, _, _, _ = deepspeed.initialize(
#     args=args,
#     model=model,
#     model_parameters=optimizer_grouped_parameters,
# )

# if dist.get_rank() == 0:
#     wandb.init(project="KPMG", name=f"{args.model_name}_NER")
    
# for epoch in range(args.epoch):
#     for train in tqdm(train_loader):
#         model.train()
#         engine.zero_grad()
#         text, label = train["data"], train["label"].cuda()
#         tokens = tokenizer(
#             text, 
#             return_tensors="pt", 
#             truncation=True, 
#             padding=True
#         )

#         input_ids = tokens.input_ids.cuda()
#         attention_mask = tokens.attention_mask.cuda()

#         output = engine.forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=label
#         )

#         loss = output.loss
#         engine.backward(loss)
#         engine.step()
#         classification_results = output.logits.argmax(-1)

#         acc = 0
#         for res, lab in zip(classification_results, label):
#             if res == lab:
#                 acc += 1
                
#     if dist.get_rank() == 0:
#         wandb.log({"loss": loss})
#         wandb.log({"epoch": epoch})
#         wandb.log({"acc": acc / len(classification_results)})

    
#     with torch.no_grad():
#         model.eval()
#         for eval in tqdm(eval_loader):
#             eval_text, eval_label = eval["data"], eval["label"].cuda()
#             eval_tokens = tokenizer(
#                 eval_text,
#                 return_tensors="pt",
#                 truncation=True,
#                 padding=True
#             )

#             eval_input_ids = eval_tokens.input_ids.cuda()
#             eval_attention_mask = eval_tokens.attention_mask.cuda()

#             eval_out = engine.forward(
#                 input_ids=eval_input_ids,
#                 attention_mask=eval_attention_mask,
#                 labels=eval_label
#             )
#             eval_loss = eval_out.loss
#             eval_classification_results = eval_out.logits.argmax(-1)
            
#             eval_acc = 0
#             for res, lab in zip(eval_classification_results, eval_label):
#                 if res == lab:
#                     eval_acc += 1
                    
#         if dist.get_rank() == 0: 
#             wandb.log({"eval_loss": eval_loss})
#             wandb.log({"eval_acc": eval_acc / len(eval_classification_results)})

        
#         ckpt_dir = f"model_save/{args.model_name.replace('/', '-')}-{epoch}-korfin"
#         model.save_pretrained(ckpt_dir)