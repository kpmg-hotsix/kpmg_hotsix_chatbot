'''
python NER_inference.py -c config.json
'''
import argparse
import csv
import os
import tarfile
from typing import List
from argparse import ArgumentParser
import json
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          PreTrainedTokenizer)
KLUE_NER_OUTPUT = "output.csv"  # the name of the output file should be output.csv

parser = ArgumentParser()
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--model_name", type=str, default="lighthouse/mdeberta-v3-base-kor-further")
config = json.load(open(parser.parse_args().config))
args = parser.parse_args()

@torch.no_grad()
def inference():
    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenzier
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    label_list = config["inference"]["label_list"]

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels=len(label_list))
    model.load_state_dict(torch.load(config["inference"]["checkpoint"], map_location="cpu"))
    model.to(device)
    model.config.id2label = {i: label for i, label in zip(range(len(label_list)), label_list)}
    model.config.label2id = {label: i for i, label in zip(range(len(label_list)), label_list)}

    with torch.no_grad():
        while True:
            t = input("\nKorean: ")
            tokens = tokenizer(
                t,
                return_tensors="pt",
                max_length=510,
                add_special_tokens=False,
                truncation=True,
                padding=True
            ).to(device)

            input_ids = tokens.input_ids.cuda()
            print(tokenizer.convert_ids_to_tokens(input_ids[0]))

            attention_mask = tokens.attention_mask.cuda()

            logits = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            predicted_token_class_ids = logits.argmax(-1)
            predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
            print(predicted_tokens_classes)

if __name__ == "__main__":
    inference()
