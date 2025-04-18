"""
File: web_scraper.py

Description: Web scraping module for collecting BBC article text from URLs.
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from datetime import datetime


def scrape_bbc_article(url):
    """
    Scrape a BBC news article.

    Parameters:
    url (str): URL of the BBC article

    Returns:
    dict: Dictionary containing article metadata and content
    """
    try:
        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make the request
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract article title
        title_element = soup.find('h1', class_='ssrcss-1pl2zfy-StyledHeading')
        title = title_element.text.strip() if title_element else "No title found"

        # Extract article date
        date_element = soup.find('time')
        date_str = date_element.get('datetime') if date_element else None

        # Try to convert date to consistent format
        if date_str:
            try:
                date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                date = date_obj.strftime('%Y-%m-%d')
            except:
                date = date_str
        else:
            date = "Unknown date"

        # Extract article body
        # BBC articles typically have content in <div> with data-component="text-block"
        article_divs = soup.find_all('div', attrs={'data-component': 'text-block'})
        paragraphs = []

        for div in article_divs:
            p_tags = div.find_all('p')
            for p in p_tags:
                paragraphs.append(p.text.strip())

        # Join paragraphs into article text
        article_text = '\n\n'.join(paragraphs)

        # Extract article source (BBC)
        source = "BBC"

        # Get article ID from URL
        article_id_match = re.search(r'([a-z0-9]+)$', url)
        article_id = article_id_match.group(1) if article_id_match else "unknown_id"

        return {
            'title': title,
            'date': date,
            'source': source,
            'url': url,
            'text': article_text,
            'id': article_id
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def scrape_multiple_articles(urls, output_dir='data/bbc_articles', delay=1):
    """
    Scrape multiple BBC articles and save them to files.

    Parameters:
    urls (list): List of BBC article URLs
    output_dir (str): Directory to save article files
    delay (int): Delay between requests in seconds

    Returns:
    list: List of successfully scraped article info dictionaries
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    scraped_articles = []

    for i, url in enumerate(urls):
        print(f"Scraping article {i + 1}/{len(urls)}: {url}")

        # Scrape the article
        article = scrape_bbc_article(url)

        if article:
            # Save article to file
            filename = f"bbc_article_{article['id']}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                # Write metadata
                f.write(f"TITLE: {article['title']}\n")
                f.write(f"DATE: {article['date']}\n")
                f.write(f"SOURCE: {article['source']}\n")
                f.write(f"URL: {article['url']}\n")
                f.write(f"ID: {article['id']}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                # Write article text
                f.write(article['text'])

            print(f"Saved article to {filepath}")
            scraped_articles.append(article)

            # Add delay between requests to be respectful
            if i < len(urls) - 1:
                time.sleep(delay)

    return scraped_articles


def main():
    # BBC articles about Depp-Heard case
    bbc_urls = [
        "https://www.bbc.com/news/articles/c977de3x007o",
        "https://www.bbc.com/news/world-us-canada-64031252",
        "https://www.bbc.com/news/world-us-canada-61673676",
        "https://www.bbc.com/news/entertainment-arts-65635628",
        "https://www.bbc.com/news/world-us-canada-61070988",
        "https://www.bbc.com/news/world-us-canada-61668780",
        "https://www.bbc.com/news/world-us-canada-61263794",
        "https://www.bbc.com/news/world-us-canada-61467766"
    ]

    scraped_articles = scrape_multiple_articles(bbc_urls)
    print(f"Successfully scraped {len(scraped_articles)} articles")


if __name__ == "__main__":
    main()