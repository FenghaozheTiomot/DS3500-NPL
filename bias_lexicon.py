"""
File: bias_lexicon.py

Description: Functions for loading and managing gender-specific lexicons.
"""

import json
import csv
import os


def load_json_lexicon(filepath):
    """
    Load a gender-specific lexicon from a JSON file

    Parameters:
    filepath (str): Path to the JSON lexicon file

    Returns:
    dict: Dictionary containing gender lexicon data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lexicon = json.load(f)
    return lexicon


def load_csv_lexicon(filepath, male_col=0, female_col=1, has_header=True):
    """
    Load a gender-specific lexicon from a CSV file

    Parameters:
    filepath (str): Path to the CSV lexicon file
    male_col (int): Column index for male terms
    female_col (int): Column index for female terms
    has_header (bool): Whether the CSV has a header row

    Returns:
    dict: Dictionary mapping male terms to female terms
    """
    lexicon = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        # Skip header if present
        if has_header:
            next(reader)

        for row in reader:
            if len(row) > max(male_col, female_col):
                male_term = row[male_col].strip().lower()
                female_term = row[female_col].strip().lower()

                if male_term and female_term:
                    lexicon[male_term] = female_term

    return lexicon


def create_basic_gender_lexicons():
    """
    Create and save basic gender lexicons to files

    Returns:
    dict: Dictionary with paths to created lexicon files
    """
    # Define basic gender word pairs
    gender_pairs = {
        "pronouns": {
            "he": "she",
            "him": "her",
            "his": "hers",
            "himself": "herself"
        },
        "nouns": {
            "man": "woman",
            "boy": "girl",
            "guy": "gal",
            "father": "mother",
            "son": "daughter",
            "husband": "wife",
            "brother": "sister",
            "uncle": "aunt",
            "nephew": "niece",
            "actor": "actress",
            "waiter": "waitress",
            "host": "hostess"
        },
        "adjectives": {
            # Stereotypically male-coded
            "strong": -1,
            "dominant": -1,
            "aggressive": -1,
            "confident": -1,
            "logical": -1,
            "rational": -1,
            "brave": -1,
            "stoic": -1,
            "ambitious": -1,
            "assertive": -1,

            # Stereotypically female-coded
            "emotional": 1,
            "nurturing": 1,
            "sensitive": 1,
            "passive": 1,
            "gentle": 1,
            "compassionate": 1,
            "empathetic": 1,
            "delicate": 1,
            "beautiful": 1,
            "pretty": 1,
            "hysterical": 1,
            "irrational": 1,
            "dramatic": 1
        }
    }

    # Create directory if it doesn't exist
    os.makedirs("lexicons", exist_ok=True)

    # Save lexicons to files
    filepaths = {}
    for lexicon_name, lexicon_data in gender_pairs.items():
        filepath = f"lexicons/gender_{lexicon_name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(lexicon_data, f, indent=2)
        filepaths[lexicon_name] = filepath

    return filepaths


def add_word_to_lexicon(lexicon_file, lexicon_type, male_word=None, female_word=None, bias_value=None):
    """
    Add a new word or word pair to an existing lexicon

    Parameters:
    lexicon_file (str): Path to lexicon file
    lexicon_type (str): Type of lexicon ('pairs' or 'values')
    male_word (str): Male word (for pairs)
    female_word (str): Female word (for pairs)
    bias_value (float): Gender bias value (for value lexicons)

    Returns:
    bool: True if successful, False otherwise
    """
    try:
        # Load existing lexicon
        with open(lexicon_file, 'r', encoding='utf-8') as f:
            lexicon = json.load(f)

        # Add new entry based on lexicon type
        if lexicon_type == 'pairs' and male_word and female_word:
            lexicon[male_word.lower()] = female_word.lower()
        elif lexicon_type == 'values' and male_word and bias_value is not None:
            lexicon[male_word.lower()] = bias_value
        else:
            return False

        # Save updated lexicon
        with open(lexicon_file, 'w', encoding='utf-8') as f:
            json.dump(lexicon, f, indent=2)

        return True
    except Exception as e:
        print(f"Error adding word to lexicon: {e}")
        return False


def merge_lexicons(filepaths, output_file):
    """
    Merge multiple lexicons into a single output file

    Parameters:
    filepaths (list): List of lexicon file paths
    output_file (str): Path for merged output file

    Returns:
    dict: Merged lexicon dictionary
    """
    merged = {}

    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lexicon = json.load(f)
                merged.update(lexicon)
        except Exception as e:
            print(f"Error loading lexicon {filepath}: {e}")

    # Save merged lexicon
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)

    return merged