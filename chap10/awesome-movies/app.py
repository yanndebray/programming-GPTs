import streamlit as st
import faiss
import pandas as pd
import numpy as np
from openai import OpenAI
import urllib.request
from pathlib import Path

# Load environment variables
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Page config
st.set_page_config(
    page_title="Movie Search Engine",
    page_icon="🎬",
    layout="wide"
)

def download_files():
    """Download necessary files from S3 if they don't exist."""
    s3_urls = {
        "movie_data.pkl": "https://programming-gpts.s3.us-east-1.amazonaws.com/movie_data.pkl",
        "movie_embeddings.faiss": "https://programming-gpts.s3.us-east-1.amazonaws.com/movie_embeddings.faiss"
    }
    
    for filename, url in s3_urls.items():
        file_path = Path(filename)
        if not file_path.exists():
            try:
                st.toast(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
                st.toast(f"Downloaded {filename}")
            except Exception as e:
                st.error(f"Error downloading {filename}: {str(e)}")
                raise

        # Verify file exists and has content
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} does not exist after download attempt")
        if file_path.stat().st_size == 0:
            raise ValueError(f"File {filename} is empty after download")
        
        # st.toast(f"File {filename} exists with size: {file_path.stat().st_size} bytes")

@st.cache_resource
def init_search():
    """Initialize search resources."""
    index = faiss.read_index("movie_embeddings.faiss")
    df = pd.read_pickle("movie_data.pkl")
    return client, index, df

# Call download_files outside of the cached function
download_files()
client, index, df = init_search()

def search_movies(query: str, search_type: str, k: int = 12) -> list:
    """
    Search for movies using the specified search type.
    
    Args:
        query: Search query text
        search_type: One of 'hybrid', 'keyword', or 'semantic'
        k: Number of results to return
    """
    if search_type == 'keyword':
        # Simple keyword search in DataFrame
        mask = df['Name'].str.contains(query, case=False) | \
               df['Description'].str.contains(query, case=False) | \
               df['Keywords'].str.contains(query, case=False)
        results_df = df[mask].head(k)
        indices = results_df.index.tolist()
    else:
        # Generate embedding for semantic search
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        query_embedding = np.array([response.data[0].embedding]).astype('float32')
        
        if search_type == 'hybrid':
            # First get keyword matches
            mask = df['Name'].str.contains(query, case=False) | \
                   df['Description'].str.contains(query, case=False) | \
                   df['Keywords'].str.contains(query, case=False)
            keyword_indices = df[mask].index.tolist()
            
            # Then get semantic matches
            _, semantic_indices = index.search(query_embedding, k)
            
            # Combine and deduplicate results
            indices = list(dict.fromkeys(keyword_indices + semantic_indices[0].tolist()))[:k]
        else:  # semantic
            _, indices = index.search(query_embedding, k)
            indices = indices[0].tolist()
    
    # Format results
    results = []
    for idx in indices:
        movie = df.iloc[idx]
        results.append({
            'movie_id': int(movie['id']),
            'title': movie['Name'],
            'year': int(movie['year']),
            'poster_link': movie['PosterLink'],
            'genres': movie['Genres'],
            'director': movie['Director']
        })
    
    return results

def display_movie_card(movie: dict, cols, idx: int, show_button: bool = True) -> None:
    """Helper function to display a movie card consistently"""
    with cols[idx % 4]:
        if movie.get('poster_link'):
            st.image(movie['poster_link'], use_container_width=True)
        st.markdown(f"**{movie['title']}** ({movie['year']})")
        st.write(f"Genres: {movie['genres']}")
        st.write(f"Director: {movie['director']}")
        if show_button:
            if st.button(
                "Show Similar Movies",
                key=f"btn_{movie['movie_id']}",
                use_container_width=True,
                type="primary"
            ):
                st.query_params["movie_id"] = movie['movie_id']
                st.rerun()

# App header
st.title("🎬 Movie Search Engine")

# Get query parameters
selected_movie_id = st.query_params.get("movie_id", None)
search_query = st.query_params.get("q", "")
search_type = st.query_params.get("type", "Hybrid")

# If a movie is selected, show it in the sidebar
if selected_movie_id is not None and selected_movie_id != "None":
    st.sidebar.subheader("Selected Movie")
    
    # Add "Back to Search" button at the top of sidebar
    if st.sidebar.button("← Back to Search", use_container_width=True):
        # Clear movie_id but keep search parameters
        st.query_params["movie_id"] = None
        st.rerun()
    
    st.sidebar.divider()  # Add visual separation
    
    selected_idx = df[df['id'] == int(selected_movie_id)].index[0]
    selected_movie = df.iloc[selected_idx]
    selected_movie_data = {
        'movie_id': int(selected_movie['id']),
        'title': selected_movie['Name'],
        'year': int(selected_movie['year']),
        'poster_link': selected_movie['PosterLink'],
        'genres': selected_movie['Genres'],
        'director': selected_movie['Director']
    }
    
    # Display selected movie in sidebar
    if selected_movie_data.get('poster_link'):
        st.sidebar.image(selected_movie_data['poster_link'], use_container_width=True)
    st.sidebar.markdown(f"**{selected_movie_data['title']}** ({selected_movie_data['year']})")
    st.sidebar.write(f"Genres: {selected_movie_data['genres']}")
    st.sidebar.write(f"Director: {selected_movie_data['director']}")
    
    # Create text for similarity search
    movie_text = f"""Title: {selected_movie['Name']}
        Description: {selected_movie['Description']}
        Plot: {selected_movie['Plot']}
        Genres: {selected_movie['Genres']}
        Director: {selected_movie['Director']}
        Actors: {selected_movie['Actors']}
        Keywords: {selected_movie['Keywords']}"""
    
    # Get similar movies
    similar_movies = search_movies(movie_text, 'semantic')
    similar_movies = [m for m in similar_movies if m['movie_id'] != selected_movie_data['movie_id']]
    
    # Display similar movies in main area
    st.subheader(f"Movies Similar to {selected_movie_data['title']}")
    cols = st.columns(4)
    for idx, movie in enumerate(similar_movies):
        display_movie_card(movie, cols, idx)

else:
    # Regular search interface
    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:
        search_query = st.text_input("Search for movies", search_query)

    with search_col2:
        search_type = st.selectbox(
            "Search type",
            ["Hybrid", "Keyword", "Semantic"],
            index=["Hybrid", "Keyword", "Semantic"].index(search_type)
        )

    # Update query parameters
    st.query_params["q"] = search_query if search_query else None
    st.query_params["type"] = search_type

    # Search results
    if search_query:
        results = search_movies(search_query, search_type.lower())
        
        if results:
            st.subheader("Search Results")
            cols = st.columns(4)
            for idx, movie in enumerate(results):
                display_movie_card(movie, cols, idx)
                
                # If this movie's button is clicked, update query parameters
                if st.session_state.get(f"btn_{movie['movie_id']}", False):
                    st.query_params["movie_id"] = movie['movie_id']
                    st.query_params["q"] = search_query if search_query else None
                    st.query_params["type"] = search_type
                    st.rerun()