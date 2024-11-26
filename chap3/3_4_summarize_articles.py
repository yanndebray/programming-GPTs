# python 3_4_summarize_articles.py "open-interpreter.txt"
from utils import bot
import argparse

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


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Extract article content from a given txt file on the path.")
    parser.add_argument("file_path", type=str, help="The file path of the article to fetch content from.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract content from the provided file path
    file_path = args.file_path
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    
    summary = summarize_article(content)
    print(summary)