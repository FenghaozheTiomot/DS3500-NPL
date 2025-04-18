"""
File: gender_bias_textastic.py

Description: A comprehensive framework for gender bias analysis in text,
combining the original Textastic functionality with gender bias analysis capabilities.
"""

from collections import Counter, defaultdict
import random as rnd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import json
import pandas as pd
import sankey as sk


class GenderBiasTextastic:
    def __init__(self):
        """ Constructor with extended functionality for gender bias analysis"""
        self.data = defaultdict(dict)
        self.gender_lexicons = {}
        self.embeddings = None
        self.gender_direction = None
        self.stop_words = set()

    def simple_text_parser(self, filename):
        """ For processing simple, unformatted text documents """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read().lower()

            # Clean and tokenize
            words = text.split()
            word_count = Counter(words)
            num_words = len(words)

            results = {
                'wordcount': word_count,
                'numwords': num_words,
                'clean_text': text,
                'tokens': words
            }

            print(f"Parsed: {filename}")
            return results
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            # Fallback with empty results
            return {
                'wordcount': Counter(),
                'numwords': 0,
                'clean_text': "",
                'tokens': []
            }

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework and
        store data extracted from the document to be used
        later in visualizations """

        results = self.simple_text_parser(filename)  # default
        if parser is not None:
            results = parser(filename)

        if label is None:
            label = filename

        # Store results in the data dictionary
        for k, v in results.items():
            self.data[k][label] = v

    def load_stop_words(self, stopfile):
        """Load common stop words to filter from analysis"""
        try:
            with open(stopfile, 'r', encoding='utf-8') as f:
                self.stop_words = set(word.strip().lower() for word in f.readlines())
            print(f"Loaded {len(self.stop_words)} stop words")
        except Exception as e:
            print(f"Error loading stop words: {e}")

    def compare_num_words(self):
        """ A simple visualization that creates a bar
        chart comparing the number of words in each file. """

        num_words = self.data['numwords']
        for label, nw in num_words.items():
            plt.bar(label, nw)
        plt.title("Word Count Comparison")
        plt.ylabel("Number of Words")
        plt.show()

    # Gender bias analysis methods

    def load_gender_lexicon(self, filename, lexicon_name):
        """Load gender-specific lexicons (male/female word pairs)"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
            self.gender_lexicons[lexicon_name] = lexicon
            print(f"Loaded gender lexicon '{lexicon_name}' with {len(lexicon)} entries")
        except Exception as e:
            print(f"Error loading gender lexicon: {e}")

    def load_word_embeddings(self, embedding_file=None):
        """Load pre-trained word embeddings"""
        try:
            import gensim.downloader as api
            print("Downloading small word embeddings...")
            self.embeddings = api.load('glove-twitter-25')  # 只有25维，更小更快
            print(f"Loaded {len(self.embeddings.index_to_key)} word vectors")
            return True
        except Exception as e:
            print(f"Error loading word embeddings: {e}")
            return False

    def compute_gender_direction(self, gender_pairs=None):
        """Compute gender direction in vector space"""
        if self.embeddings is None:
            print("Word embeddings must be loaded first")
            return None

        if not gender_pairs:
            # Default gender pairs like in the Word2Vec paper
            gender_pairs = [('he', 'she'), ('man', 'woman'), ('boy', 'girl'),
                            ('father', 'mother'), ('husband', 'wife'), ('male', 'female')]

        print(f"Computing gender direction using {len(gender_pairs)} word pairs...")
        direction_vectors = []
        for male, female in gender_pairs:
            if male in self.embeddings and female in self.embeddings:
                # Male - Female to get gender direction
                direction = self.embeddings[male] - self.embeddings[female]
                # Normalize
                direction = direction / np.linalg.norm(direction)
                direction_vectors.append(direction)

        if not direction_vectors:
            print("No valid gender pairs found in embeddings")
            return None

        # Average gender direction
        self.gender_direction = np.mean(direction_vectors, axis=0)
        return self.gender_direction

    def project_word_on_gender_axis(self, word):
        """Project word onto gender axis"""
        if self.embeddings is None or self.gender_direction is None:
            return None

        if word not in self.embeddings:
            return None

        # Compute projection
        projection = np.dot(self.embeddings[word], self.gender_direction)
        return projection

    def get_words_by_gender_score(self, min_count=5, exclude_stopwords=True):
        """Get all words with their gender scores, filtered by frequency"""
        if self.embeddings is None or self.gender_direction is None:
            print("Word embeddings and gender direction must be computed first")
            return {}

        # Combine word counts from all texts
        all_words = Counter()
        for src, counts in self.data.get('wordcount', {}).items():
            all_words.update(counts)

        # Filter by frequency and stopwords
        filtered_words = {}
        for word, count in all_words.items():
            if count >= min_count and word in self.embeddings:
                if exclude_stopwords and word in self.stop_words:
                    continue
                score = self.project_word_on_gender_axis(word)
                if score is not None:
                    filtered_words[word] = score

        return filtered_words

    def analyze_gendered_associations(self, entity, context_window=5):
        """Analyze gendered associations for a specific entity in texts"""
        if self.embeddings is None or self.gender_direction is None:
            print("Word embeddings and gender direction must be computed first")
            return {}

        results = {}

        for src, tokens in self.data.get('tokens', {}).items():
            # Find instances of entity
            entity_indices = [i for i, token in enumerate(tokens) if token.lower() == entity.lower()]

            # Get context words around entity
            context_words = []
            for idx in entity_indices:
                start = max(0, idx - context_window)
                end = min(len(tokens), idx + context_window + 1)
                context_words.extend(tokens[start:end])

            # Calculate gender bias for context words
            gender_scores = {}
            for word in set(context_words):
                if word in self.stop_words:
                    continue
                if word in self.embeddings:
                    score = self.project_word_on_gender_axis(word)
                    if score is not None:
                        gender_scores[word] = score

            results[src] = gender_scores

        return results

    def wordcount_sankey(self, word_list=None, k=5):
        """
        Generate a Sankey diagram connecting texts to words.

        Parameters:
        word_list (list): Optional list of specific words to include
        k (int): Number of top words to include from each text if word_list not provided
        """
        # Check if wordcount data exists
        if 'wordcount' not in self.data:
            print("No wordcount data available")
            return

        # Get words to include in diagram
        if word_list is None:
            # Collect top-k words from each text
            word_list = set()
            for src, counts in self.data['wordcount'].items():
                # Filter out stop words and get top k
                top_words = [word for word, _ in counts.most_common(k * 3)
                             if word not in self.stop_words][:k]
                word_list.update(top_words)

            word_list = list(word_list)

        # Build dataframe for Sankey
        sankey_data = []
        for src, counts in self.data['wordcount'].items():
            for word in word_list:
                count = counts.get(word, 0)
                if count > 0:
                    sankey_data.append({'source': src, 'target': word, 'value': count})

        # Convert to DataFrame
        df = pd.DataFrame(sankey_data)

        # Generate Sankey if data exists
        if not df.empty:
            sk.make_sankey(df, ['source', 'target'], 'value')
        else:
            print("No data available for Sankey diagram")

    def gender_word_association_sankey(self, entity1='depp', entity2='heard', top_n=10):
        """
        Generate Sankey diagram showing gendered word associations with two entities

        Parameters:
        entity1 (str): First entity name (e.g., 'depp')
        entity2 (str): Second entity name (e.g., 'heard')
        top_n (int): Number of top associated words to include
        """
        # Analyze gendered associations for both entities
        entity1_assoc = self.analyze_gendered_associations(entity1)
        entity2_assoc = self.analyze_gendered_associations(entity2)

        # Compile all words and scores across sources
        entity1_words = Counter()
        entity2_words = Counter()

        for src, scores in entity1_assoc.items():
            for word, score in scores.items():
                entity1_words[word] += self.data['wordcount'][src].get(word, 0)

        for src, scores in entity2_assoc.items():
            for word, score in scores.items():
                entity2_words[word] += self.data['wordcount'][src].get(word, 0)

        # Get top words for each entity
        top_entity1_words = [w for w, _ in entity1_words.most_common(top_n)]
        top_entity2_words = [w for w, _ in entity2_words.most_common(top_n)]

        # Build data for Sankey diagram
        sankey_data = []

        # Add connections between entity1 and its words
        for word in top_entity1_words:
            if word in self.embeddings:
                gender_score = self.project_word_on_gender_axis(word)
                if gender_score is not None:
                    sankey_data.append({
                        'source': entity1,
                        'target': word,
                        'value': entity1_words[word],
                        'gender_score': gender_score
                    })

        # Add connections between entity2 and its words
        for word in top_entity2_words:
            if word in self.embeddings:
                gender_score = self.project_word_on_gender_axis(word)
                if gender_score is not None:
                    sankey_data.append({
                        'source': entity2,
                        'target': word,
                        'value': entity2_words[word],
                        'gender_score': gender_score
                    })

        # Convert to DataFrame
        df = pd.DataFrame(sankey_data)

        # Generate Sankey if data exists
        if not df.empty:
            print(f"Generating gender association Sankey diagram for {entity1} and {entity2}")
            sk.make_sankey(df, ['source', 'target'], 'value')
        else:
            print("No data available for gender association Sankey diagram")

    def gender_bias_projection_plot(self, words_list=None, top_n=100):
        """
        Plot word projections along gender axis, similar to the Word2Vec visualization

        Parameters:
        words_list (list): Optional specific list of words to plot
        top_n (int): Number of top frequent words to include if words_list not provided
        """
        if self.embeddings is None or self.gender_direction is None:
            print("Word embeddings and gender direction must be computed first")
            return

        if not words_list:
            # Take most frequent words from corpus
            all_words = Counter()
            for src, counts in self.data.get('wordcount', {}).items():
                all_words.update(counts)

            # Filter stop words
            words_list = [w for w, c in all_words.most_common(top_n * 3)
                          if w not in self.stop_words][:top_n]

        # Get embeddings for these words
        word_vectors = []
        valid_words = []
        for word in words_list:
            if word in self.embeddings:
                word_vectors.append(self.embeddings[word])
                valid_words.append(word)

        if not valid_words:
            print("No valid words found in embeddings")
            return

        # Apply PCA for dimensionality reduction (like in the Word2Vec paper)
        pca = PCA(n_components=2)
        result = pca.fit_transform(word_vectors)

        # Calculate projection onto gender direction
        gender_scores = [self.project_word_on_gender_axis(word) for word in valid_words]

        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(result[:, 0], result[:, 1], c=gender_scores, cmap='coolwarm',
                              alpha=0.7, s=100)

        # Add labels for important words
        for i, word in enumerate(valid_words):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]),
                         xytext=(5, 2), textcoords='offset points',
                         fontsize=10)

        plt.colorbar(scatter, label='Gender Bias Score (+ masculine, - feminine)')
        plt.title('Word Embeddings Projected on Gender Direction')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        plt.tight_layout()
        plt.show()

    def gender_bias_comparison_subplots(self, entities=['depp', 'heard']):
        """
        Generate subplots comparing gender bias across media sources for specific entities

        Parameters:
        entities (list): List of entity names to analyze
        """
        if self.embeddings is None or self.gender_direction is None:
            print("Word embeddings and gender direction must be computed first")
            return

        # Get all sources
        sources = list(self.data.get('wordcount', {}).keys())

        if not sources:
            print("No text sources available")
            return

        # Set up subplot grid
        n_sources = len(sources)
        n_cols = min(3, n_sources)
        n_rows = (n_sources + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Analyze each entity's associations
        entity_associations = {}
        for entity in entities:
            entity_associations[entity] = self.analyze_gendered_associations(entity)

        # Create plots for each source
        for i, source in enumerate(sources):
            ax = axes[i]

            # Collect data for this source
            source_data = []
            for entity in entities:
                if source in entity_associations[entity]:
                    # Get words and their gender scores
                    words_scores = entity_associations[entity][source]

                    # Sort by absolute score to get most biased words
                    sorted_words = sorted(words_scores.items(),
                                          key=lambda x: abs(x[1]), reverse=True)[:10]

                    for word, score in sorted_words:
                        source_data.append({
                            'entity': entity,
                            'word': word,
                            'score': score
                        })

            # Convert to DataFrame for plotting
            if source_data:
                df = pd.DataFrame(source_data)

                # Group by entity
                for entity, group in df.groupby('entity'):
                    # Sort by score for better visualization
                    group = group.sort_values('score')

                    # Plot horizontal bars
                    bars = ax.barh(group['word'], group['score'],
                                   alpha=0.7, label=entity)

                # Add reference line at 0
                ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

                # Set labels and title
                ax.set_title(f"Gender Bias in {source}")
                ax.set_xlabel("Gender Score (+ masculine, - feminine)")

                # Add legend
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"No data for {source}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

        # Hide empty subplots
        for i in range(len(sources), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()