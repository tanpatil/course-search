import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pickle
import torch
import streamlit as st

@st.cache
def load_data(datafile_path = "./data/courses.csv"):
	df = pd.read_csv(datafile_path)
	df = df.dropna()
	return df

@st.cache(allow_output_mutation=True)
def store_embeddings(embeddings, sentences):
    with open('./data/embeddings_dbert.pkl', "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

@st.cache(allow_output_mutation=True)
def load_embeddings():
    #Load sentences & embeddings from disc
	with open('./data/embeddings_dbert.pkl', "rb") as fIn:
		stored_data = pickle.load(fIn)
		stored_sentences = stored_data['sentences']
		stored_embeddings = stored_data['embeddings']
	return stored_sentences, stored_embeddings

def main():
    
    # ENCODING SENTENCES
    tqdm.pandas()
    df = load_data()
    df['combined'] = "Name: " + df.name.str.strip() + "; Description: " + df.description.str.strip()
    
    # convert df['combined'] to array of strings
    sentences = df['combined'].values.astype('U')
    store_embeddings([], sentences)
    
    # RUN MODEL
    model = SentenceTransformer('multi-qa-distilbert-cos-v1')
    embeddings = model.encode(sentences, show_progress_bar=True)
    
    store_embeddings(embeddings, sentences)
    
    corpus, corpus_embeddings  = load_embeddings()
    print(corpus_embeddings.shape)

    
    
    

if __name__ == "__main__":
    main()
    
    