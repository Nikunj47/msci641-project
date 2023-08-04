#!/usr/local/bin/python3
import argparse
import json
import pandas as pd
import re 
import numpy as np
import torch
import transformers 
from transformers import BertForQuestionAnswering, BertTokenizer, AutoModel, AutoTokenizer,RobertaForQuestionAnswering,T5ForConditionalGeneration
from transformers import pipeline
from tqdm import tqdm
from transformers import T5Tokenizer, T5Model

class Qa_model:
    def __init__(self, model_name, num_answers = 1):
        self.model_name = model_name
        self.num_answers = num_answers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_name, local_files_only=True)
        self.model = pipeline("question-answering", self.model_name, tokeniezer = self.tokenizer, max_length=500, truncation=True, return_overflowing_tokens=True, stride = 128, top_k= self.num_answers)
                     #pipeline("question-answering", "deepset/roberta-base-squad2", tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2"), max_length=500
                      #              , truncation=True, return_overflowing_tokens=True, stride=doc_stride, top_k=5)
        # self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # self.t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

    def predict(self, question, context):
        answer = self.model(question = question, context = context)
        # inputs = self.t5_tokenizer.encode("summarize: " + answer['answer'], return_tensors="pt", max_length=512, truncation=True)
        # outputs = self.t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        # return self.t5_tokenizer.decode(outputs[0])
        return answer


def get_phrase(row, model_phrase):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    return [model_phrase.predict(question, context)['answer']]
    # return [model_phrase.predict(question, context)]


def get_passage(row, model_passage):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    answer = model_passage.predict(question,context)['answer']
    # answer = model_passage.predict(question,context)

    candidates = []
    for sentence in context.split('.'):
        if answer in sentence:
            candidates.append(sentence.strip())
    
    if not candidates:
        print('No candidates found')
        return ['']
    elif len(candidates) == 1:
        return [candidates[0]]
    elif len(candidates) > 1:
        print('Multiple candidates found')
        return [candidates[0]]


def get_multi(row, model_multi):
    question = row.get('postText')[0]
    context = ' '.join(row.get('targetParagraphs'))

    current_context = context
    results = []
    try:
        for _ in range(0,5):
            #current_context = current_context
            candidates = model_multi.predict(question, current_context)[0]
            current_result = candidates['answer']
            results.append(current_result)
            current_context = re.sub(current_result, '', current_context)
    except:
        print("Error generating multipart spoiler")
        results = ["Error"]
    return results
    

def predict(inputs, model_phrase, model_passage, model_multi):
    for row in tqdm(inputs):
        if row.get('spoilerType') == 'phrase':
            answer = get_phrase(row, model_phrase)

        elif row.get('spoilerType') == 'passage':
            answer = get_passage(row, model_passage)
        
        elif row.get('spoilerType') == 'multi':
            answer = get_multi(row, model_multi)
        else:
            print("Tag not found")
            raise NotImplemented

        yield {'id': row['id'], 'spoiler': answer}


def run_baseline(input_file, output_file, model_phrase, model_passage, model_multi):
    with open(input_file, 'r') as inp, open(output_file, 'w') as out:
         inp = [json.loads(i) for i in inp]

         for output in predict(inp, model_phrase, model_passage, model_multi):
            out.write(json.dumps(output) + '\n')


if __name__ == '__main__':

    model_phrase = Qa_model("../models/roberta-base-squad2")
    model_passage = Qa_model("../models/roberta-base-squad2")
    model_multi = Qa_model("../models/roberta-base-squad2", 5)
    # args = parse_args()
    input = '../data/input/test-merged.jsonl'
    output = '../data/output/spoiler-predictions.jsonl'
    run_baseline(input, output, model_phrase = model_phrase, model_passage = model_passage, model_multi = model_multi)
