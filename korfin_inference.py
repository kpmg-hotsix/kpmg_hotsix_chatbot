"""
python interactive.py
"""

from argparse import ArgumentParser
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "jinmang2/kpfbert"
ckpt_name = "model_save/jinmang2-kpfbert-4-korfin/pytorch_model.bin"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.load_state_dict(torch.load(ckpt_name, map_location="cpu"))
model.cuda()

while True:
    t = input("\nTitle: ")
    tokens = tokenizer(
        t,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()

    output = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    classification_results = output.logits.argmax(-1)
    result = str(classification_results.item())
    result = result.replace("0", "NEGATIVE").replace("1", "NEUTRAL").replace("2", "POSITIVE")
    print(f"Result: {result}")
    # print(f"Result: {'True' if classification_results.item() else 'False'}")