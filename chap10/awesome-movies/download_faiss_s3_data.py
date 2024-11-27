import urllib.request

pkl = "https://programming-gpts.s3.us-east-1.amazonaws.com/movie_data.pkl"
faiss = "https://programming-gpts.s3.us-east-1.amazonaws.com/movie_embeddings.faiss"

urllib.request.urlretrieve(pkl, "movie_data.pkl")
urllib.request.urlretrieve(faiss, "movie_embeddings.faiss")