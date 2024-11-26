from utils import extract_content, format_content, summarize_article, classify_article
import json

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

if __name__ == "__main__":
    bookmarks = [
        {
            "title": "open-interpreter",
            "url": "https://simonwillison.net/2024/Nov/24/open-interpreter/",
            "date_added": "2024/11/24",
        },
        {
            "title": "Le Chat",
            "url": "https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-etonnant-chatgpt-a-la-francaise-de-mistral-ai_6219436_4408996.html",
            "date_added": "2024/03/01",
        }
    ]
    processed_articles = process_bookmarks(bookmarks)
    with open("processed_articles.json", "w") as f:
        json.dump(processed_articles, f, indent=2)
