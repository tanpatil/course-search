## Course Directory for UCSD

Semantic search of UCSD courses built BERT pre-trained models and Pinecone vector database.

![Triton Spanner Page](screenshot.png)

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
