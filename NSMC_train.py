import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
import random
import os
# model = AutoModelForSequenceClassification.from_pretrained("mdeberta-v3-base-kor-further")  # DebertaV2ForModel

random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = AutoModelForSequenceClassification.from_pretrained(
    'lighthouse/mdeberta-v3-base-kor-further',
    num_labels=2,
).cuda()

tokenizer = AutoTokenizer.from_pretrained("lighthouse/mdeberta-v3-base-kor-further")  # DebertaV2Tokenizer (SentencePiece)

train_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
train_data = train_data.dropna(axis=0)
train_data = train_data[:100000]
train_text, train_labels = (
    train_data["document"].values,
    train_data["label"].values,
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(train_text, train_labels)
]

train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

eval_data = pd.read_csv("data/ratings_train.txt", delimiter="\t")
eval_data = eval_data.dropna(axis=0)
eval_data = eval_data[100000:]
eval_text, eval_labels = (
    eval_data["document"].values,
    eval_data["label"].values
)

dataset = [
    {"data": t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]

eval_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
)

optimizer = AdamW(params=model.parameters(),
    lr=3e-5, weight_decay=3e-7
)

epochs = 10
for epoch in range(epochs):
    model.train()
    for train in tqdm(train_loader):
        optimizer.zero_grad()
        text, label = train["data"], train["label"].cuda()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            # max_length=140

        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )

        loss = output.loss
        loss.backward()        
        optimizer.step()
        classification_results = output.logits.argmax(-1)

        acc = 0
        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

    print({"loss": loss.item()})
    print({"acc": acc / len(classification_results)})   

    with torch.no_grad():
        model.eval() 
        for eval in tqdm(eval_loader):
            eval_text, eval_label = eval["data"], eval["label"].cuda()
            eval_tokens = tokenizer(
                eval_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                # max_length=140
            )
            input_ids = eval_tokens.input_ids.cuda()
            attention_mask = eval_tokens.attention_mask.cuda()

            eval_out = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=eval_label
            )

            eval_classification_results = eval_out.logits.argmax(-1)
            eval_loss = eval_out.loss

            eval_acc = 0
            for res, lab in zip(eval_classification_results, eval_label):
                if res == lab:
                    eval_acc += 1

        print({"eval_loss": eval_loss.item()})  
        print({"eval_acc": eval_acc / len(eval_classification_results)}) 
        print({"epoch": epoch+1})
        torch.save(model.state_dict(), f"model_save/BERT_fintuing_NSMC-{epoch+1}.pt")