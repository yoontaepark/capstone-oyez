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

## se3 lib
import spacy
from pysbd.utils import PySBDFactory
from numpy.linalg import norm


# define args 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["Justia_summary", "Justia_holding", "facts_of_the_case", "question", "conclusion"])
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--se3", action="store_true")
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


## se3 functions
# cos sim 
def cos_sim(ref_sent, sent):

    # to compare, we convert each sentences into list 
    # note that as sentences will be sent to torch, 
    # we convert into numpy 
    list_a = ref_sent.detach().cpu().numpy()
    list_b = sent.detach().cpu().numpy()

    # calculate based on the cosine similarity formula 
    cos_sim_score = np.dot(list_a,list_b)/((norm(list_a)*norm(list_b)))

    # return cosine similarity score 
    return cos_sim_score

# algo 2
def semantic_sim(sentence, curr_chunk, next_chunk):
    
    # initialize cos sim scores for both curr and next chunk 
    score_curr_chunk = 0
    score_next_chunk = 0
    
    # for each sentences in current chunk, add cos_sim score
    for enc_chunk_sentence in curr_chunk: 
      score_curr_chunk += cos_sim(enc_chunk_sentence, sentence)

    # also, for each sentences in next chunk, add cos_sim score
    for enc_chunk_sentence in next_chunk: 
      score_next_chunk += cos_sim(enc_chunk_sentence, sentence)

    # then we divide into each size of chunk to get an average score 
    score_curr_chunk /= len(curr_chunk)
    score_next_chunk /= len(next_chunk)  

    # compare the score, and append to the chunk where the score is higher 
    if score_curr_chunk > score_next_chunk:
      curr_chunk.append(sentence)
    else:
      next_chunk.append(sentence)

# algo 3
# objective of this function is to create a target chunk and distribute sentences based on main_chunk
def target_assign(summary_sentences, main_chunk):

    # initialize targets, rouge score
    targets = [[] for _ in range(len(main_chunk))]
    rouge = evaluate.load('rouge')

    # as all documents would not have string format of the sentence, we change sentences into string format
    # i.e. word -> 'word' 
    summary_sentences = [str(input) for input in summary_sentences]  

    # then for each target sentences, check rouge score for each of main chunk
    for sentence in summary_sentences:
        rouge_scores = []

    # for each chunk, we calculate rouge score and append to the list
        for c in main_chunk:

            chunk_score = rouge.compute(predictions=[c], references=[sentence])
            rouge_scores.append(chunk_score['rougeL']) # rougeL, this can be changed to any evaluation metric

            # then we select the highest rouge score index and append to the target chunk index 
            idx = np.argmax(rouge_scores)
            targets[idx].append(sentence)

    # after that, we merge list of sentences to one paragraph 
    for i in range(len(targets)):
        targets[i] = " ".join(str(label) for label in targets[i])

    # return target chunks 
    return targets      

# algo 1: Note that we seperate algo 3 in this function
# this happens as we don't know test set 
def create_chunk(document, lower_size, upper_size, target=None):
    # assigning chunks 
    main_chunk = []
    curr_chunk = []
    next_chunk = []

    # also setting lower / upper size
    lower_size = lower_size
    upper_size = upper_size
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    # To compare similarity score, we encode document 
    doc_str_converted = [str(input) for input in document]  
    enc_doc_str_converted = tokenizer(doc_str_converted, max_length=tokenizer.model_max_length, padding="longest", truncation=True, return_tensors='pt')

    # main function
    for enc_sentence in enc_doc_str_converted['input_ids']:
    
        # fill min sentences for current chunk, and next chunk 
        if len(curr_chunk) + 1 < lower_size:
            curr_chunk.append(enc_sentence)
        elif len(next_chunk) + 1 < lower_size:
            next_chunk.append(enc_sentence)

        # if any of the chunks are filled, move to main chunk
        elif len(curr_chunk) + 1 > upper_size:
            main_chunk.append(curr_chunk)
            curr_chunk = []
        elif len(next_chunk) + 1 > upper_size:
            main_chunk.append(next_chunk)
            next_chunk = []        

        # for rest of the sentences, conduct similarity comparison and append sentence to 
        # either current or next chunk
        else:
            semantic_sim(enc_sentence, curr_chunk, next_chunk)

    # safety code: this prevents the case where curr_chunk and next_chunk are piled up but not added to the main chunk 
    # because they didn't reached the upper size  
    if len(curr_chunk) != 0:
        main_chunk.append(curr_chunk)
    if len(next_chunk) != 0:
        main_chunk.append(next_chunk)

    ## this makes decoded chunks of main_chunk 
    decoded_main_chunk = [tokenizer.batch_decode(chunk, skip_special_tokens=True) for chunk in main_chunk] 

    ## converting seperated sentences into chunk 
    for i in range(len(decoded_main_chunk)):
        decoded_main_chunk[i] = " ".join(str(label) for label in decoded_main_chunk[i])

    ## this creates either input chunk or (input,output) pair of chunks 
    # this is generating only chunk
    if target == None:
        return decoded_main_chunk

    # this is generating (input,output) pair of chunks 
    else:
        # this gives a list of target sentences distributed to target chunks 
        target_chunks = target_assign(target, decoded_main_chunk)

        ## this only returns decodecd chunk and target chunk pair that exists
        filtered_main_chunk = []
        filtered_target_chunk = []

        for i in range(len(target_chunks)):
            if target_chunks[i] != '':
                filtered_main_chunk.append(decoded_main_chunk[i])
                filtered_target_chunk.append(target_chunks[i])

        # return filtered (input, target) chunk pairs 
        return filtered_main_chunk, filtered_target_chunk

# preprocssing. this encodes input (and labels)  
def preprocess_function_testset(df):
    # we only encode input 
    model_inputs = tokenizer(df["input"], max_length=tokenizer.model_max_length, truncation=True)        
    return model_inputs

# decoding function, this converts list of intergers into sentence 
def decode_data_testset(prediction):
    decoded_preds = tokenizer.batch_decode(prediction, skip_special_tokens=True)

    # We convert back into the sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]

    return decoded_preds    


# main function     
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
        print('2-1. Finetuning...')

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
        
    elif args.se3:
        print('2-2. se3...')

        ### 2-2: 1) here we divide train, test dataset first 
        train_df = df_task.sample(frac=0.8,random_state=0).reset_index().drop(columns='index')
        test_df = df_task.drop(train_df.index).reset_index().drop(columns='index')
        
        
        ### 2-2: 2) preprocessing of train dataset
        # divide into list of sentences from entire doc by using pysbd
        # this applies to train/test input, and train target set 
        # creating a spacy variable 
        nlp = spacy.blank('en')
        nlp.add_pipe('sentencizer')

        # this converts a paragraph into list of sentences 
        train_df['input_pysbd'] = train_df['input'].apply(lambda x: list(nlp(x).sents))
        train_df['target_pysbd'] = train_df['target'].apply(lambda x: list(nlp(x).sents))
        
        # this is the main function to run
        input_chunks_list = []
        target_chunks_list = []

        # for train set, we iterate every doc-summary pair and create chunk-target pair 
        for i in range(len(train_df)):
            input_chunks, target_chunks = create_chunk(train_df['input_pysbd'][i], lower_size=10, upper_size=50, target = train_df['target_pysbd'][i])
            
            input_chunks_list.extend(input_chunks)
            target_chunks_list.extend(target_chunks)
        
        # then we convert into dataframe 
        df = pd.DataFrame(list(zip(input_chunks_list, target_chunks_list)), columns =['input', 'target'])
        
        ### 2-2: 3) training train dataset (by spliting again into train - eval set)
        # define your model
        model_name = 'facebook/bart-base'  # bart-base
        tokenizer_name = 'facebook/bart-base' # bart-base
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        dataset = Dataset.from_pandas(df)
        train_test_df = dataset.train_test_split(test_size=0.2)

        # preprocess dataset
        tokenized_datasets = train_test_df.map(preprocess_function, batched=True)

        # finetune the model
        train = tokenized_datasets["train"]
        test = tokenized_datasets["test"]
        
        # clear cache 
        gc.collect()
        torch.cuda.empty_cache()
        
        # train the model
        trainer = model_train(train, test)
        
        ### 2-2: 4) preprocessing test dataset 
        # divide into list of sentences from entire doc by using pysbd
        # this applies to test input set 
        nlp = spacy.blank('en')
        nlp.add_pipe('sentencizer')
        test_df['input_pysbd'] = test_df['input'].apply(lambda x: list(nlp(x).sents))
        
        
        ### 2-2: 5) predicting test dataset 
        # for each test input, predict summary based on generated chukns 
        decoded_preds_merged_list = []
        for i in range(len(test_df)):

            input_chunks_list = []
            input_chunks = create_chunk(test_df['input_pysbd'][i], lower_size=10, upper_size=50)

            input_chunks_list.extend(input_chunks)

            df = pd.DataFrame(input_chunks_list, columns = ['input'])
            dataset = Dataset.from_pandas(df)

            # preprocess dataset
            tokenized_datasets = dataset.map(preprocess_function_testset, batched=True)

            # predict: if 10 chunks -> 10 summary 
            predictions = trainer.predict(tokenized_datasets, max_length=tokenizer.model_max_length)

            # decode prediction result 
            decoded_preds = decode_data_testset(predictions[0])

            decoded_preds_merged =  " ".join(decoded_preds)

            decoded_preds_merged_list.append(decoded_preds_merged)        
        
        ### 2-2: 6) adding pred column that contains predicted summary 
        # convert dataset to dataframe and add prediction column 
        test_df['pred'] = decoded_preds_merged_list
        
        # evaluate and print the result
        result = evaluate_result(test_df)
        print(result)
        print()
        
        # this is just for consistencey of entire code 
        # this replaces original df_task 
        df_task = test_df

        
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
    
    file_name = os.getcwd() + '/result_1116.csv' # your path
    df_task.to_csv(file_name, index=False)
    print('Done!')
    
    