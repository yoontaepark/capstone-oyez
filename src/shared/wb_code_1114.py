## python 'your_file.py' 'task_name' : baseline
## python 'your_file.py' 'task_name' --finetune : finetuning  

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import pickle5 as pickle
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import evaluate

import os 
import argparse
import re
from bs4 import BeautifulSoup

## finetune lib
import transformers
import torch
import nltk
nltk.download('punkt')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

## holding lib
from nltk.tokenize import sent_tokenize
import gc

# define args 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["Justia_summary", "Justia_holding", "facts_of_the_case", "question", "conclusion"])
    parser.add_argument("--finetune", action="store_true")
    return parser.parse_args()

# cleaning html tags
def clean_html(raw_html):
    cleantext = BeautifulSoup(raw_html, "html.parser").text
    cleantext = re.sub('\xa0', ' ', cleantext)
    cleantext = re.sub('\n', ' ', cleantext)
    return cleantext

# generate dataset
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
    
    # Data cleaning 
    # remove html
    task_df[task_name] = task_df[task_name].apply(lambda x: clean_html(x))

    # remove wrong values
    current_unk = ['Annotation', 'Currently unknown.', 'Currently unavailable.', '']
    task_df = task_df[~task_df[task_name].isin(current_unk)]

    # filter columns that are needed 
    task_df = task_df[['file_name_oyez', 'Justia_txt', task_name]]
    task_df.columns = ['id', 'input', 'target']

    print('Done!')
    print('Shape of dataset: ', task_df.shape)
    print()
    
    return task_df    

# preprocssing. this encodes input (and labels)  
def preprocess_function(df):

    model_inputs = tokenizer(df["input"], max_length=tokenizer.model_max_length, truncation=True)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    labels = tokenizer(df["target"], max_length=tokenizer.model_max_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# training function, returns the model (which is trainer)
def model_train(train_df, test_df):
    # define args for finetuning
    batch_size = 8
    args = Seq2SeqTrainingArguments(
        f"{model_name}-wb-finetuned-1114",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10, # or 20
        predict_with_generate=True,
        # fp16=True,
        # push_to_hub=True,
    )

    # collator fixed of max length 
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_df,
        eval_dataset=test_df,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer    

# decoding function, this converts list of intergers into sentence 
def decode_data(prediction, label):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # We convert back into the sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    return decoded_preds, decoded_labels


## pipeline for bart-base
def run_model(model_name, tokenizer_name, df):

    print('2. Model running...')
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, framework='pt', device=0) # add device=0
    df['pred'] = summarizer(df['input'].values.tolist(), max_length=tokenizer.model_max_length, truncation=True) # removed min_length
    print('Done!')
    print()
    
    return df

# evaluate by rouge score, we focus on rougeL
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
    
    # generate task
    df_task = generate_task(file_path, args.task)
    
    # define your model
    model_name = 'facebook/bart-base'  # bart-base
    tokenizer_name = 'facebook/bart-base' # bart-base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # finetune the model 
    if args.finetune:
        print('2. Finetuning...')

        # clear cache 
        gc.collect()
        torch.cuda.empty_cache()
        
        ## In case of Finetuning 
        # convert df to Dataset(huggingface) class 
        dataset = Dataset.from_pandas(df_task)
        train_test_df = dataset.train_test_split(test_size=0.2)

        # preprocess dataset
        tokenized_datasets = train_test_df.map(preprocess_function, batched=True)

        # finetune the model
        train_df = tokenized_datasets["train"]
        test_df = tokenized_datasets["test"]
        trainer = model_train(train_df, test_df)

        # predict
        predictions, label_ids, _ = trainer.predict(test_df, max_length=tokenizer.model_max_length)

        # decode prediction result 
        decoded_preds, decoded_labels = decode_data(predictions, label_ids)

        # convert dataset to dataframe and add prediction column 
        df_task = train_test_df['test'].to_pandas()
        df_task['pred'] = decoded_preds

        print('Done!')

    # else, just use the model that we imported 
    else:
        df_task = run_model(model_name, tokenizer_name, df_task)
        df_task['pred'] = df_task['pred'].apply(lambda x: x['summary_text'])

    # evaluate and print the result
    result = evaluate_result(df_task)
    print(result)
    print()
      
    ## check some examples 
    print('4. Examples: ')
    for i in range(5):
        print('case:', i) 
        print('prediction: ', df_task.loc[i, 'pred'])
        print()
        print('actual: ', df_task.loc[i, 'target'])
        print()

    # save checkpoint
    print('5. Saving result file...')
    
    file_name = os.getcwd() + '/result_1114.csv' # your path
    df_task.to_csv(file_name, index=False)
    print('Done!')
    
    