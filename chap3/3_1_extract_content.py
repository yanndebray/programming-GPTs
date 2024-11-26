# python 3_1_extract_content.py "https://simonwillison.net/2024/Nov/24/open-interpreter/"
import requests
from bs4 import BeautifulSoup
import argparse
from utils import bot

def extract_content(url):
    """
    Extracts the title and main content of an article from the given URL.
    
    Args:
        url (str): The URL of the article to extract content from.
        
    Returns:
        tuple: A tuple containing the article's title and content as strings.
    """
    try:
        # Fetch the URL content
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        html = response.text
        
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else "No Title Found"
        content = soup.get_text(separator='\n')
        
        return title, content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None


def format_content(title, content):
    """
    Formats the article content using OpenAI's API.
    
    Args:
        title (str): The title of the article.
        content (str): The main content of the article.
        
    Returns:
        str: A formatted and summarized version of the content.
    """
    try:
        prompt = f"""You are an expert text formatter and summarizer.
        Here is an article:
        
        Title: {title}
        
        Content:
        {content}
        
        Please format this scraped text into a clean, readable content."""
        formatted_content = bot(prompt)

        return formatted_content
    except Exception as e:
        print(f"Error formatting content with OpenAI: {e}")
        return "Failed to format content."
    

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract and print article content from a given URL.")
    parser.add_argument("url", type=str, help="The URL of the article to fetch content from.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract content from the provided URL
    url = args.url
    title, content = extract_content(url)
    formatted_content = format_content(title, content)
    with open(f"{title}.txt","w", encoding="utf-8") as f:
        f.write(formatted_content)
        
    # Print the result
    if title and content:
        print(f"Title: {title}")
        print("\nContent:\n")
        print(content)
    else:
        print("Failed to extract content.")
