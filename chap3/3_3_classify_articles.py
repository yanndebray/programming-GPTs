# python 3_3_classify_articles.py "open-interpreter.txt"
from utils import bot
import argparse

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
    
    category = classify_article(content)
    print(category)