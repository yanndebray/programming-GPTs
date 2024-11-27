import pandas as pd
import numpy as np
import faiss
import os
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pickle
import urllib.request
# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download movie_data.csv and wiki_movie_plots_deduped.csv from S3 if they don't exist
s3_urls = {
    "movie_data.csv": "https://programming-gpts.s3.us-east-1.amazonaws.com/movie_data.csv",
    "wiki_movie_plots_deduped.csv": "https://programming-gpts.s3.us-east-1.amazonaws.com/wiki_movie_plots_deduped.csv"
}

for filename, url in s3_urls.items():
    file_path = os.path.join("data", filename)
    if not os.path.exists(file_path):
        try:
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, file_path)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {str(e)}")
            raise

    # Verify file exists and has content
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {filename} does not exist after download attempt")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File {filename} is empty after download")


# Load the same data as in add_data.py
df = pd.read_csv("data/movie_data.csv", 
                usecols=['id', 'Name', 'PosterLink', 'Genres', 'Actors', 
                        'Director','Description', 'DatePublished', 'Keywords'], 
                parse_dates=["DatePublished"])

df["year"] = df["DatePublished"].dt.year.fillna(0).astype(int)
df.drop(["DatePublished"], axis=1, inplace=True)
df = df[df.year > 1970]

# Load and merge plot data
plots = pd.read_csv("data/wiki_movie_plots_deduped.csv")
plots = plots[plots['Release Year'] > 1970]
plots = plots[plots.duplicated(subset=['Title', 'Release Year', 'Plot']) == False]
plots = plots[plots.duplicated(subset=['Title', 'Release Year']) == False]
plots = plots[['Title', 'Plot', 'Release Year']]
plots.columns = ['Name', 'Plot', 'year']

# Merge datasets
df = df.merge(plots, on=['Name', 'year'], how='left').fillna('')
df.reset_index(drop=True, inplace=True)

# Create text for embedding
def create_embedding_text(row):
    return f"""Title: {row['Name']}
    Description: {row['Description']}
    Plot: {row['Plot']}
    Genres: {row['Genres']}
    Director: {row['Director']}
    Actors: {row['Actors']}
    Keywords: {row['Keywords']}"""

# Generate embeddings
embeddings = []
batch_size = 100  # Process in batches to avoid rate limits

for i in tqdm(range(0, len(df), batch_size)):
    batch = df.iloc[i:i+batch_size]
    texts = [create_embedding_text(row) for _, row in batch.iterrows()]
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    
    batch_embeddings = [embedding.embedding for embedding in response.data]
    embeddings.extend(batch_embeddings)

# Convert embeddings to numpy array
embeddings_array = np.array(embeddings).astype('float32')

# Create FAISS index
dimension = len(embeddings_array[0])  # Should be 1536 for text-embedding-3-small
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

# Save the index and dataframe
faiss.write_index(index, "movie_embeddings.faiss")
df.to_pickle("movie_data.pkl")

print(f"Created FAISS index with {len(df)} movies") 