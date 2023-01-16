## course search for UCSD 

semantic search of UCSD courses initially made for personal use. 

built with BERT pre-trained models and Pinecone (vector database). 

to use
```
pip install -r requirements
streamlit run app.py
```

to scrape and retrain model
```
python3 scraper.py 
python3 embeddings.py
```
make sure you have a Pinecone account and store your key in .env

note: scraped data was not thoroughly validated and there could be some name/description mismatches