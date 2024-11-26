import openai
import requests
from bs4 import BeautifulSoup

def bot(prompt, temperature=0):
    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()


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


def translate_article(content):
    prompt = f"""
    Translate the following article in English:

    \"\"\"
    {content}
    \"\"\"

    Translation:
    """
    translation = bot(prompt)
    return translation.strip()


def classify_article(content):
    prompt = """
    You are an AI assistant that classifies technical articles into categories. The categories are:

    - Machine Learning
    - Web Development
    - Mobile Development
    - Cloud Computing
    - Data Science
    - Programming Languages
    - Other

    For example:

    Article:
    \"\"\"
    An introduction to Kubernetes and container orchestration.
    \"\"\"
    Category: Cloud Computing

    Article:
    \"\"\"
    Understanding React Hooks and their usage in modern web apps.
    \"\"\"
    Category: Web Development

    Now classify this article:

    \"\"\"
    {}
    \"\"\"

    Category:""".format(content)
    category = bot(prompt)
    return category.strip()


def summarize_article(content):
    key_points_prompt = f"""
    You are an AI assistant that summarizes technical articles. Read the article below and think through the main points step by step before writing the final summary.

    Article:
    \"\"\"
    {content}
    \"\"\"

    First, outline the key points and main ideas of the article. Then, write a concise summary incorporating these points.

    Key Points:
    """
    # First, get the key points
    key_points_response = bot(key_points_prompt)
    key_points = key_points_response.strip()

    # Now, generate the final summary using the key points
    summary_prompt = f"""
    Using the key points below, write a concise summary of the article.

    Key Points:
    {key_points}

    Summary:
    """
    summary = bot(summary_prompt)
    return summary.strip()


def process_bookmarks(bookmarks):
    processed_articles = []
    for bookmark in bookmarks:
        url = bookmark['url']
        title, content = extract_content(url)
        content = format_content(title, content)
        summary = summarize_article(content)
        category = classify_article(content)
        article_data = {
            'title': title or bookmark['title'],
            'category': category,
            'summary': summary,
        }
        processed_articles.append(article_data)
        # Optionally save to a database or file
    return processed_articles