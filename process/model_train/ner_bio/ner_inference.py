import os
import json
from argparse import ArgumentParser
import torch
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from ner_dataloader import DataloaderForNER

def define_args():
    p = ArgumentParser()

    p.add_argument('--config', type=str, required=True)
    p.add_argument('--model_name', type=str, default="lighthouse/mdeberta-v3-base-kor-further")
    p.add_argument('--max_length', type=int, default=510)
    p.add_argument('--label_type', type=str, default=None)
    inference_config = json.load(open(p.parse_args().config))
    config = p.parse_args()
    return config, inference_config

@torch.no_grad()
def inference(config, inference_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataloader = DataloaderForNER(config.model_name, max_length=config.max_length, label_type=config.label_type)
    label_to_id = dataloader.label_to_id

    model = AutoModelForTokenClassification.from_pretrained(config.model_name, num_labels=len(label_to_id))
    model.load_state_dict(torch.load(inference_config["inference"]["checkpoint"], map_location="cpu"))
    model.to(device)
    model.config.label_to_id = dataloader.label_to_id
    model.config.id_to_label = {i: label for label, i in model.config.label_to_id.items()}
    with torch.no_grad():
        while True:
            text = input("Input text: ")
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=config.max_length,
                add_special_tokens=False,
                truncation=True,
                padding=True
            ).to(device)

            input_ids = tokens.input_ids.cuda()
            attention_mask = tokens.attention_mask.cuda()
            logits = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).logits
            predicted_token_class_ids = logits.argmax(-1)
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            predicted_tokens_classes = [model.config.id_to_label[token.item()] for token in predicted_token_class_ids[0]]
            print(list(zip(input_tokens, predicted_tokens_classes)))

if __name__ == "__main__":
    config, inference_config = define_args()
    inference(config, inference_config)