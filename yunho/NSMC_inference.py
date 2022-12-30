import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
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

ckpt_name = "model_save/BERT_fintuing_NSMC-1.pt"
model = AutoModelForSequenceClassification.from_pretrained(
    'lighthouse/mdeberta-v3-base-kor-further',
    num_labels=2,
)
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

tokenizer = AutoTokenizer.from_pretrained("lighthouse/mdeberta-v3-base-kor-further")  # DebertaV2Tokenizer (SentencePiece)

eval_data = pd.read_csv("data/ratings_test.txt", delimiter="\t")
eval_data = eval_data.dropna(axis=0)
# eval_data = eval_data[:300]
eval_text, eval_labels = (
    eval_data["document"].values,
    eval_data["label"].values,
)

dataset = [
    {"data": tokenizer.cls_token + t, "label": l}
    for t, l in zip(eval_text, eval_labels)
]

eval_loader = DataLoader(
    dataset,
    batch_size=128,
    num_workers=8,
    drop_last=True,
    pin_memory=True,
    
)

with torch.no_grad():
    model.eval()
    acc = 0
    for data in tqdm(eval_loader):
        text, label = data["data"], data["label"].cuda()
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        input_ids = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()
        # label = torch.tensor(label).cuda()

        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        classification_results = output.logits.argmax(-1).cuda()

        for res, lab in zip(classification_results, label):
            if res == lab:
                acc += 1

    print(f"acc: {acc / len(dataset)}")