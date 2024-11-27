import urllib.request
import os

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