import pinecone
import os
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sentence_transformers import SentenceTransformer, util
import time
import streamlit as st
import re
from PIL import Image

load_dotenv()
key = os.getenv("PINECONE_KEY")

pinecone.init(api_key=key, environment="us-west1-gcp")
im = Image.open("site.png")
st.set_page_config(
		page_title="Triton Search",
		page_icon=im)

def setup():
	pinecone.create_index("courses", dimension=768, metric="cosine", pod_type="p1")

def insert_pinecone(arr):
	index = pinecone.Index("courses")
	for i in tqdm(range(len(arr))):
		index.upsert([(str(i), arr[i])])

@st.cache(allow_output_mutation=True)
def store_hashtable(arr):
	# create a hashtable of the course id and the course name
	hashtable = {}
	for i in range(len(arr)):
		converted = ""
		if arr[i].find("Prerequisites") == -1:
			converted = re.sub(r"Name: (.+?); Description: (.+)", r"**\1**:\n \2", arr[i])
		else:
			converted = re.sub(r"Name: (.+?); Description: (.+) Prerequisites: (.+)", r"**\1**:\n \2\n **Prerequisites:** \3", arr[i])
		hashtable[str(i)] = converted
	# store hashtable into binary
	with open('./data/hashtable.pickle', 'wb') as handle:
		pickle.dump(hashtable, handle, protocol=pickle.HIGHEST_PROTOCOL)

@st.cache(allow_output_mutation=True)
def load_hashtable():
	with open('./data/hashtable.pickle', 'rb') as handle:
		hashtable = pickle.load(handle)
	return hashtable

def search_pinecone(query, topk, model):
	index = pinecone.Index("courses")

	tic = time.perf_counter()
	embedding = model.encode(query)
	toc = time.perf_counter()
	print(f"Time to encode query: {toc - tic:0.4f} seconds")

	tic = time.perf_counter()
	results = index.query(embedding.tolist(), top_k=topk, include_values=True)
	toc = time.perf_counter()
	print(f"Time to query: {toc - tic:0.4f} seconds")

	return results

@st.cache(allow_output_mutation=True)
def load_embeddings():
    #Load sentences & embeddings from disc
	with open('./data/embeddings_dbert.pkl', "rb") as fIn:
		stored_data = pickle.load(fIn)
		stored_sentences = stored_data['sentences']
		stored_embeddings = stored_data['embeddings']
	return stored_sentences, stored_embeddings

@st.cache(allow_output_mutation=True)
def load_model():
	model = SentenceTransformer('multi-qa-distilbert-cos-v1')
	return model

def main(): 

	tqdm.pandas()
	
	# setup() # uncomment to create index

	# corpus, embeddings = load_embeddings() # uncomment to load embeddings from binary
	
	# insert_pinecone(embeddings.tolist()) # uncomment to insert into pinecone

	# store_hashtable(corpus) # uncomment to store hashtable into binary

	tic = time.perf_counter()
	model = load_model()
	toc = time.perf_counter()
	print(f"Time to load model: {toc - tic:0.4f} seconds")

	hashtable = load_hashtable() 

	# Streamlit	
	st.title("Triton Courses")
	st.write("[Semantic search](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview) lets you search with meaning - try 'cooking' or 'animal doctor'.")
	st.write("[github](https://github.com/punnkam/ucsd-courses), [twitter](https://twitter.com/punnkam)")

	c1, c2 = st.columns((12, 1))
    
    # user input
	with c1:
		user_input = st.text_input(label="Search box", placeholder="quantum AI shape rotating cryptography")
    
    # filters
	with c2:
		num_results = st.text_input("Results", 10)
	
	if user_input:
		results = search_pinecone(user_input, int(num_results), model)
		for result in results['matches']:
			# write results to streamlit
			st.write(hashtable[result['id']])
		
if __name__ == "__main__":
	main()
