"""
File: bias_analysis.py

Description: Functions for analyzing gender bias in text using word embeddings
and vector projection techniques.
"""

import numpy as np
from collections import Counter
import pandas as pd


def find_entity_contexts(tokens, entity, window_size=5):
    """
    Extract context windows around mentions of an entity

    Parameters:
    tokens (list): List of tokens from a document
    entity (str): Target entity to find contexts for
    window_size (int): Size of context window on each side

    Returns:
    list: List of context words around entity mentions
    """
    # Find all indices where entity appears
    entity_indices = [i for i, token in enumerate(tokens)
                      if token.lower() == entity.lower()]

    # Extract context windows
    contexts = []
    for idx in entity_indices:
        start = max(0, idx - window_size)
        end = min(len(tokens), idx + window_size + 1)
        # Add all context words except the entity itself
        contexts.extend([tokens[i] for i in range(start, end) if i != idx])

    return contexts


def calculate_gender_bias_score(word, embeddings, gender_direction):
    """
    Calculate gender bias score by projecting word onto gender direction

    Parameters:
    word (str): Word to analyze
    embeddings: Word embedding model
    gender_direction (numpy.array): Gender direction vector

    Returns:
    float: Gender bias score (positive = masculine, negative = feminine)
    """
    if word not in embeddings or gender_direction is None:
        return None

    # Project word vector onto gender direction
    projection = np.dot(embeddings[word], gender_direction)
    return projection


def analyze_entity_gender_bias(corpus, entity, embeddings, gender_direction, stop_words=None):
    """
    Analyze gender bias in words associated with an entity

    Parameters:
    corpus (dict): Dictionary mapping source names to token lists
    entity (str): Entity to analyze
    embeddings: Word embedding model
    gender_direction (numpy.array): Gender direction vector
    stop_words (set): Set of stop words to exclude

    Returns:
    pd.DataFrame: DataFrame with gender bias analysis results
    """
    if stop_words is None:
        stop_words = set()

    results = []

    for source, tokens in corpus.items():
        # Get context words around entity mentions
        contexts = find_entity_contexts(tokens, entity)

        # Count occurrences of each context word
        context_counts = Counter(contexts)

        # Calculate gender bias for each context word
        for word, count in context_counts.items():
            if word in stop_words or len(word) < 3:
                continue

            if word in embeddings:
                bias_score = calculate_gender_bias_score(word, embeddings, gender_direction)

                if bias_score is not None:
                    results.append({
                        'source': source,
                        'entity': entity,
                        'word': word,
                        'count': count,
                        'bias_score': bias_score
                    })

    # Convert to DataFrame
    return pd.DataFrame(results)


def compute_entity_bias_summary(bias_df, entity):
    """
    Compute summary statistics for entity gender bias

    Parameters:
    bias_df (pd.DataFrame): DataFrame with bias analysis results
    entity (str): Entity to summarize

    Returns:
    dict: Dictionary with summary statistics
    """
    # Filter for specified entity
    entity_df = bias_df[bias_df['entity'] == entity]

    if entity_df.empty:
        return {'entity': entity, 'data_available': False}

    # Calculate weighted average bias score (weighted by word count)
    total_count = entity_df['count'].sum()
    weighted_bias = (entity_df['bias_score'] * entity_df['count']).sum() / total_count

    # Count male-associated and female-associated words
    male_biased = entity_df[entity_df['bias_score'] > 0]['count'].sum()
    female_biased = entity_df[entity_df['bias_score'] < 0]['count'].sum()

    # Most biased words in each direction
    most_male_biased = entity_df.sort_values('bias_score', ascending=False).head(5)
    most_female_biased = entity_df.sort_values('bias_score', ascending=True).head(5)

    return {
        'entity': entity,
        'data_available': True,
        'weighted_bias_score': weighted_bias,
        'male_biased_count': male_biased,
        'female_biased_count': female_biased,
        'male_female_ratio': male_biased / female_biased if female_biased > 0 else float('inf'),
        'most_male_biased_words': list(zip(most_male_biased['word'], most_male_biased['bias_score'])),
        'most_female_biased_words': list(zip(most_female_biased['word'], most_female_biased['bias_score']))
    }


def compare_entity_bias(bias_df, entity1, entity2):
    """
    Compare gender bias between two entities

    Parameters:
    bias_df (pd.DataFrame): DataFrame with bias analysis results
    entity1 (str): First entity
    entity2 (str): Second entity

    Returns:
    dict: Dictionary with comparison results
    """
    summary1 = compute_entity_bias_summary(bias_df, entity1)
    summary2 = compute_entity_bias_summary(bias_df, entity2)

    if not summary1['data_available'] or not summary2['data_available']:
        return {'comparison_available': False}

    # Calculate bias difference
    bias_diff = summary1['weighted_bias_score'] - summary2['weighted_bias_score']

    # Find shared words and compare their bias
    entity1_words = set(bias_df[bias_df['entity'] == entity1]['word'])
    entity2_words = set(bias_df[bias_df['entity'] == entity2]['word'])
    shared_words = entity1_words.intersection(entity2_words)

    # For shared words, compare bias directly
    shared_word_bias = []
    if shared_words:
        for word in shared_words:
            e1_bias = bias_df[(bias_df['entity'] == entity1) & (bias_df['word'] == word)]['bias_score'].iloc[0]
            e2_bias = bias_df[(bias_df['entity'] == entity2) & (bias_df['word'] == word)]['bias_score'].iloc[0]
            shared_word_bias.append((word, e1_bias, e2_bias, e1_bias - e2_bias))

    return {
        'comparison_available': True,
        'bias_difference': bias_diff,
        'shared_words': shared_word_bias,
        entity1: summary1,
        entity2: summary2
    }