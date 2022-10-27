## run this code by python 'your_code' 'task_name'
## i.e. python wb_code_1027.py Justia_summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import pickle5 as pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate

import os 
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["Justia_summary", "Justia_holding", "facts_of_the_case", "question", "conclusion"])
    return parser.parse_args()

def generate_task(file_path, task_name):
    
    print('1. Generating data...')
    # open file 
    oyez = pickle.load(open(file_path, "rb"))
    
    # drop duplicates
    oyez = oyez.drop_duplicates()
    
    # creating some conditions to check non-null counts
    input = ~oyez['Justia_txt'].isnull()
    task = ~oyez[task_name].isnull()
    
    # generating dataset for task1
    task_df = oyez[input&task]
    
    # filter columns that are needed 
    task_df = task_df[['file_name_oyez', 'Justia_txt', task_name]]
    task_df.columns = ['id', 'input', 'target']
    
    print('Done!')
    print()
    
    return task_df

## pipeline for bart-base
def run_model(model_name, tokenizer_name, df):

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print('2. Model running...')
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework='pt', device=0) # add device=0
    df['pred'] = summarizer(df['input'].values.tolist(), min_length=512, max_length=1024, truncation=True)
    print('Done!')
    print()
    
    return df

# results are not good as for now 
def evaluate_result(df):

    print('3. Evalating...')
    rouge = evaluate.load('rouge')
    result = rouge.compute(predictions=df['pred'].values.tolist(), references=df['target'].values.tolist())
    
    print('Done!')
    return result

if __name__ == "__main__":
    # generate dataset for given task
    file_path = os.getcwd() + '/NYU_Oyez_data.pkl' #change file path
    
    # this is to interactively change your task and test 
    args = parse_args()
    
    df_task1 = generate_task(file_path, args.task)

    # run your model
    model_name = 'facebook/bart-base'  # bart-base
    tokenizer_name = 'facebook/bart-base' # bart-base
    df_task1 = run_model(model_name, tokenizer_name, df_task1)
    df_task1['pred'] = df_task1['pred'].apply(lambda x: x['summary_text'])

    # evaluate and print the result
    result = evaluate_result(df_task1)
    print(result)
    print()
    
    ## check some examples 
    print('4. Examples: ')
    for i in range(5):
      print('case:', i) 
      print('prediiction: ', df_task1.loc[i, 'pred'])
      print('actual: ', df_task1.loc[i, 'target'])
      print()

    # save checkpoint
    print('5. Saving result file...')
    file_name = os.getcwd() + '/result_' + args.task + '_1026.csv' # your path
    df_task1.to_csv(file_name, index=False)
    print('Done!')
