from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import csv
import codecs
import pandas as pd
import json
#copied
from pythainlp import word_tokenize
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import semantic_search_faiss
import pandas as pd
import numpy as np
import requests
from typing import Dict
from utils import NERWangchanBERTaInferenceModel
import gdown
import os

app = FastAPI()

# Assume home.html is in the same directory as main.py
current_directory = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=current_directory)

@app.get('/{input}')
async def snomed(request:Request,input:str):
    text = str(input)
    dataframe = pd.DataFrame({"Text":[text]})
    print(f'DataFrame: {dataframe}')
    result_df = pd.DataFrame()
    print(f'Blank result_df: {result_df}')


    for i in range(len(dataframe)) :
        query = dataframe['Text'].iloc[i]
        temp_df = full_pipeline(query)
        if len(temp_df) != 0 :
            temp_df['Text'] = query
            temp_df['id'] = i
            result_df = pd.concat([result_df, temp_df])
        else:
            result_df = pd.DataFrame({"id":"-","Text":"-","query":"-","conceptId":"-","term":"-","type":"-"})
        
    result_df = result_df[['id', 'Text', 'query', 'conceptId', 'term', 'type']]
    print(result_df['id'].iloc[0])
    print(result_df['Text'].iloc[0])
    print(result_df['query'].iloc[0])
    print(result_df['conceptId'].iloc[0])
    print(result_df['term'].iloc[0])
    print(result_df['type'].iloc[0])

    DOGS = []
    for i in range(len(result_df)):
        to_put =  {
                    "id":i+1,
                    "Text":result_df['Text'].iloc[i],
                    "query":result_df['query'].iloc[i],
                    "conceptId":result_df['conceptId'].iloc[i],
                    "term":result_df['term'].iloc[i],
                    "type":result_df['type'].iloc[i]
                  }
        DOGS.append(to_put)

    return templates.TemplateResponse("home.html", {"request":request, "name":"SNOMED-CT", "dogs":DOGS})


# Download
def download_file():
    print('file downloading...')
    files = {
        'bilingual_term_concept_id.csv': 'https://drive.google.com/file/d/1vFgQtxa4ddli_8-OFLtgF_Sh7DDg44zJ/view?usp=drive_link',
        'snomed_bilingual_e5_large_int8.npy': 'https://drive.google.com/file/d/1GLe-SrlxQdIqbp89ndV4dnz4gAlJ28MH/view?usp=drive_link',
        'concept_cat.csv': 'https://drive.google.com/file/d/1liE9MmgEciXPcyIJn0939MzdxGwfZ2lp/view?usp=drive_link',
        'bilingual_corpus.csv': 'https://drive.google.com/file/d/1rjQFApRQVw6oNlo2kL-yuHO0BPS8Qab0/view?usp=drive_link',
    }
    for file, url in files.items():
        if not os.path.isfile(file):
            gdown.download(url=url, output=file, fuzzy=True)

def model_load() :
    print('model downloading...')
    ## load model
    # tokenizer = AutoTokenizer.from_pretrained("BotnoiNLPteam/SNOMED-CT-Model-NER-V.1", token =  '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', cache_dir = './model')
    # model = AutoModelForTokenClassification.from_pretrained("BotnoiNLPteam/SNOMED-CT-Model-NER-V.1", token = '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', cache_dir = './model').to('cpu')
    # embedding_model = SentenceTransformer("/mnt/hdd/tone/model/multilingual-e5-large").to('cpu')
    tokenizer = AutoTokenizer.from_pretrained("BotnoiNLPteam/SNOMED-CT-Model-NER-V.1", token =  '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', cache_dir = './model')
    model = AutoModelForTokenClassification.from_pretrained("BotnoiNLPteam/SNOMED-CT-Model-NER-V.1", token = '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', cache_dir = './model').to('cpu')
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")
    inference_model = NERWangchanBERTaInferenceModel(
    model,
    tokenizer,
    idx_to_class=idx_to_class,
    class_to_idx=class_to_idx)
    return inference_model, embedding_model


classes_list = ['O', 'B-Concept', 'I-Concept']
class_to_idx: Dict[str, int] = {class_name: idx for idx, class_name in enumerate(classes_list)}
idx_to_class: Dict[int, str] = {idx: class_name for class_name, idx in class_to_idx.items()}

# Load models and download files
download_file()
inference_model, embedding_model = model_load()

# Load data
def load_data() :
    db = pd.read_csv("./bilingual_term_concept_id.csv")
    cat_df = pd.read_csv("./concept_cat.csv")
    corpus_embeddings = np.load('./snomed_bilingual_e5_large_int8.npy')
    abbreviation_map = {'ca' : 'cancer'}
    corpus_df = pd.read_csv('./bilingual_corpus.csv')
    return db, cat_df, corpus_embeddings, corpus_embeddings, abbreviation_map, corpus_df

db, cat_df, corpus_embeddings, corpus_embeddings, abbreviation_map, corpus_df = load_data()
corpus = list(corpus_df['term'])
corpus_precision = "ubinary"


# NER
def preprocess(word):
    # word tokenize
    word = word_tokenize(word)

    # convert " " to "<_>"
    word = [c.replace("_", "<_>").replace(" ", "<_>").lower() for c in word]
    return word

def extract_term(tokens, tags) :
    b_idx = []
    result = []
    for i in range(len(tags)) :
        if tags[i] == 1:
            b_idx.append(i)
    for i in range(len(b_idx)) :
        start_idx = b_idx[i]
        if i != len(b_idx)-1 :
            end_idx = b_idx[i+1]
            amount_of_o = tags[start_idx:end_idx].count(0)
            result.append(tokens[start_idx:end_idx-amount_of_o])
        else :
            amount_of_o = tags[start_idx:].count(0)
            result.append(tokens[start_idx:len(tokens)-amount_of_o])
    result = ["".join(c) for c in result]
    return result

def ner_pipeline(text):
    tokens = preprocess(text)
    tags = inference_model.predict(tokens)
    result = extract_term(tokens, tags)
    result = [c.replace("<_>", " ").strip() for c in result]
    return result


# Retrieval

def k_search(queries, embedding_model, k = 1):
    corpus_index = None
    query_embeddings = embedding_model.encode(queries, normalize_embeddings=True)
    results, search_time, corpus_index = semantic_search_faiss(
    query_embeddings,
    corpus_index=corpus_index,
    corpus_embeddings=corpus_embeddings if corpus_index is None else None,
    corpus_precision=corpus_precision,
    top_k=10,
    # calibration_embeddings=full_corpus_embeddings,
    rescore=corpus_precision != "float32",
    rescore_multiplier=4,
    exact=True,
    output_index=True,
    )
    corpus_list = [[corpus[entry['corpus_id']]for entry in result][:k] for result in results]
    return pd.DataFrame({'query':queries, 'term' : corpus_list}).explode('term')

def term_to_concept_id(term, df, term_col, concept_id_col) :
    df = df[df[term_col]==term]
    return list(df[concept_id_col])

def convert_abbreviation(text) :
    result = []
    text_split = text.split()
    for t in text_split :
        if t.lower() in abbreviation_map :
            result.append(abbreviation_map[t.lower()])
        else :
            result.append(t)
    return " ".join(result)

def check_by_api(conceptId) :
    url = "https://browser.ihtsdotools.org/snowstorm/snomed-ct/browser/MAIN/concepts/" + str(conceptId)
    headers = {
        "User-Agent": "Mozilla/5.0 "
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return data['active']
    else:
        return False
    
def find_cat(df, concept_id_column) :
    exploded_df = df.explode(concept_id_column)
    exploded_df['active'] = exploded_df['matched_concept_id'].apply(check_by_api)
    exploded_df = exploded_df[exploded_df['active']==True]
    exploded_df = exploded_df.merge(cat_df, left_on = concept_id_column, right_on = 'conceptId')
    return exploded_df

def retrieval_pipeline(result):
    result = [convert_abbreviation(c) for c in result]
    df = k_search(result, embedding_model = embedding_model, k = 1)
    df['matched_concept_id'] = df['term'].apply(lambda x : term_to_concept_id(x, db, 'term', 'conceptId'))
    df = find_cat(df, 'matched_concept_id')
    return df.rename(columns = {'term_y' : 'term', 'string' : 'type'})[['query', 'conceptId', 'term', 'type']]

def full_pipeline(text) :
    result = ner_pipeline(text)
    if len(result) == 0 :
        return pd.DataFrame()
    df = retrieval_pipeline(result)
    return df

def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

# text = "ถ่ายดำมาหลายวันแล้วค่ะ"
# dataframe = pd.DataFrame({"Text":[text]})
# dataframe = dataframe.head(2)
# print(f'DataFrame: {dataframe}')
# result_df = pd.DataFrame()
# print(f'Blank result_df: {result_df}')
# for i in range(len(dataframe)) :
#     query = dataframe['Text'].iloc[i]
#     temp_df = full_pipeline(query)
#     if len(temp_df) != 0 :
#         temp_df['Text'] = query
#         temp_df['id'] = i
#         result_df = pd.concat([result_df, temp_df])
    
# result_df = result_df.reset_index(drop = True)[['id', 'Text', 'query', 'conceptId', 'term', 'type']]
# print(result_df['id'].iloc[0])
# print(result_df['Text'].iloc[0])
# print(result_df['query'].iloc[0])
# print(result_df['conceptId'].iloc[0])
# print(result_df['term'].iloc[0])
# print(result_df['type'].iloc[0])

