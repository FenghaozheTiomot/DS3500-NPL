"""
File: main.py

Description: Main script for running the gender bias analysis of
BBC articles about the Amber Heard and Johnny Depp case.
"""

from gender_bias_textastic import GenderBiasTextastic
import web_scraper
import bias_lexicon
import os
import re
from collections import Counter
import gensim.downloader as api


def bbc_article_parser(filename):
    """Custom parser for BBC article files created by the web scraper"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split metadata from article text
        parts = content.split("=" * 50, 1)
        if len(parts) == 2:
            metadata_text, article_text = parts
        else:
            article_text = content

        # Clean and process article text
        article_text = article_text.lower()
        # Remove punctuation and numbers
        article_text = re.sub(r'[^\w\s]', '', article_text)
        article_text = re.sub(r'\d+', '', article_text)
        
        # Tokenize and filter
        tokens = article_text.split()
        
        # Custom stop words specific to legal/court reporting
        custom_stop_words = {
            'said', 'year', 'time', 'court', 'trial', 'case', 'lawyer', 
            'judge', 'evidence', 'statement', 'alleged', 'testimony',
            'according', 'would', 'could', 'also', 'like', 'even', 'may'
        }
        
        # Filter tokens
        tokens = [t for t in tokens 
                 if len(t) > 2 
                 and t not in custom_stop_words
                 and not t.startswith('http')]
        
        # Count words
        word_count = Counter(tokens)

        # Count mentions of key entities
        depp_mentions = sum(1 for t in tokens if t == 'depp' or t == 'johnny')
        heard_mentions = sum(1 for t in tokens if t == 'heard' or t == 'amber')

        return {
            'wordcount': word_count,
            'numwords': len(tokens),
            'clean_text': article_text,
            'tokens': tokens,
            'depp_mentions': depp_mentions,
            'heard_mentions': heard_mentions
        }
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return {
            'wordcount': Counter(),
            'numwords': 0,
            'clean_text': "",
            'tokens': [],
            'depp_mentions': 0,
            'heard_mentions': 0
        }


def download_glove_embeddings():
    """Download and return GloVe embeddings using gensim downloader"""
    print("Downloading GloVe word embeddings (this may take a moment)...")
    try:
        # Try to load larger model first, fall back to smaller if needed
        try:
            embeddings = api.load('glove-wiki-gigaword-300')  # 300维，基于维基百科
        except:
            embeddings = api.load('glove-twitter-200')  # 200维，比25维更好
            
        print(f"Successfully loaded embeddings with {len(embeddings)} words")
        return embeddings
    except Exception as e:
        print(f"Error downloading embeddings: {e}")
        return None


def main():
    # BBC article URLs
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

    # Ensure directories exist
    os.makedirs('data/bbc_articles', exist_ok=True)
    os.makedirs('lexicons', exist_ok=True)

    # Create basic gender lexicons if they don't exist
    if not os.path.exists('lexicons/gender_adjectives.json'):
        print("Creating basic gender lexicons...")
        bias_lexicon.create_basic_gender_lexicons()

    # Scrape articles if needed
    article_dir = 'data/bbc_articles'
    if not os.listdir(article_dir):
        print("Scraping BBC articles...")
        web_scraper.scrape_multiple_articles(bbc_urls, article_dir)

    # Initialize the framework
    gbt = GenderBiasTextastic()

    # Create and load stopwords if needed
    stopwords_file = 'data/stopwords.txt'
    if not os.path.exists(stopwords_file):
        print("Creating stopwords file...")
        os.makedirs('data', exist_ok=True)
        with open(stopwords_file, 'w', encoding='utf-8') as f:
            stopwords = ['a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else',
                         'when', 'at', 'from', 'by', 'for', 'with', 'about', 'against',
                         'between', 'into', 'through', 'during', 'before', 'after',
                         'above', 'below', 'to', 'of', 'in', 'on', 'off', 'over',
                         'under', 'again', 'further', 'then', 'once', 'here', 'there',
                         'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                         'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                         'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
                         'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                         'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
                         'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan',
                         'shouldn', 'wasn', 'weren', 'won', 'wouldn', 'i', 'me', 'my',
                         'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
                         'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                         'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                         'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                         'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                         'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                         'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                         'doing', 'as', 'until', 'while', 'of', 'out', 'said', 'says']
            f.write('\n'.join(stopwords))

    # Load stopwords
    gbt.load_stop_words(stopwords_file)

    # Load gender lexicons
    gbt.load_gender_lexicon('lexicons/gender_adjectives.json', 'adjectives')

    # Process articles
    print("Loading articles into framework...")
    for filename in os.listdir(article_dir):
        if filename.endswith('.txt'):
            article_id = filename.replace('.txt', '')
            gbt.load_text(os.path.join(article_dir, filename), article_id, parser=bbc_article_parser)

    # Download embeddings using gensim
    embeddings = download_glove_embeddings()
    if embeddings:
        # Set the embeddings in the GenderBiasTextastic object
        gbt.embeddings = embeddings

        # Compute gender direction vector
        gbt.compute_gender_direction()

        # Generate visualizations with stricter parameters
        print("Generating visualizations with stricter parameters...")

        # 1. Text-to-Word Sankey diagram (only show top 5 words per article)
        print("1. Generating Text-to-Word Sankey diagram...")
        gbt.wordcount_sankey(k=5)

        # 2. Gender-specific Sankey diagram with score threshold
        print("2. Generating Gender-specific Sankey diagram...")
        gbt.gender_word_association_sankey(entity1='depp', entity2='heard', top_n=5)

        # 3. Word projections on gender axis with higher min_count
        print("3. Generating Gender Bias Projection Plot...")
        gbt.gender_bias_projection_plot(top_n=50)  # Show fewer but more significant words

        # 4. Comparison of gender bias across articles
        print("4. Generating Gender Bias Comparison subplots...")
        gbt.gender_bias_comparison_subplots()
    else:
        print("Failed to load word embeddings. The analysis cannot continue.")


if __name__ == "__main__":
    main()