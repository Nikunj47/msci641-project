# Import the required files

import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification



def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data in jsonl format).', required=True)
    return parser.parse_args()

def load_input(df):
    with open(df, 'r') as inp:
         inp = [json.loads(i) for i in inp]
    return pd.DataFrame(inp)


class ClassificationModel:
    def __init__(self):
        self.models = {
            "passage": AutoModelForSequenceClassification.from_pretrained("../models/{model}", num_labels=2,local_files_only=True),
            "phrase": AutoModelForSequenceClassification.from_pretrained("../models/{model}", num_labels=2,local_files_only=True),
            "multi": AutoModelForSequenceClassification.from_pretrained("../models/{model}", num_labels=2,local_files_only=True)
        }
        self.tokenizer = AutoTokenizer.from_pretrained("../models/{model}",local_files_only=True)
        self.fields = {"passage": ['postText', 'targetTitle'], "phrase": ['postText'], "multi": ['postText', 'targetTitle', 'targetParagraphs']}

    def get_text(self, row, tag):
        text = ""
        for field in self.fields[tag]:
            if isinstance(row[field], list):
                text += ' '.join(row[field])
            elif isinstance(field, str):
                text += row[field]
            else:
                raise NotImplemented
        return text

    def predict_probability(self, text: str, model):
        tokenized = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**tokenized).logits
        return logits.argmax()

    def predict_one(self, row: str):
        probabilities = {}
        for tag_name, model in self.models.items():
            text = self.get_text(row, tag_name)
            probability = self.predict_probability(text, model)
            probabilities[tag_name] = probability

        return max(probabilities, key=probabilities.get)


def predict(file_path):
    df = load_input(file_path)
    uuids = list(df['id'])

    classifyer = ClassificationModel()

    for idx, i in tqdm(df.iterrows()):
        spoiler_type = classifyer.predict_one(i)
        yield {'id': uuids[idx], 'spoilerType': spoiler_type}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    print(args.input)
    output = '../data/output/predictions.jsonl'
    run_baseline(args.input, output)