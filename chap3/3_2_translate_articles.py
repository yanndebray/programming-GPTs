# python 3_2_translate_articles.py "https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-etonnant-chatgpt-a-la-francaise-de-mistral-ai_6219436_4408996.html"
from utils import bot, extract_content, format_content
import argparse

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
    translation = translate_article(content)
    with open(f"{title}.txt","w", encoding="utf-8") as f:
        f.write(translation)
    # Print the result
    if title and content:
        print(f"Title: {title}")
        print("\nContent:\n")
        print(translation)
    else:
        print("Failed to extract content.")