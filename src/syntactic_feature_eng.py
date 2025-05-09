# Description: This script contains functions for syntactic feature engineering,
#  including visualizing dependency trees, calculating syntactic complexity,
#  and using a pre-trained language model to calculate word probabilities.

import pandas as pd
import json
import spacy
import networkx as nx
import pygraphviz as pgv
from IPython.display import Image, display
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import sys

nlp = spacy.load("en_core_web_sm") 

def visualize_dependency_tree(text):
    """
    Visualize the dependency tree for a given text using spaCy's displacy.
    """
    doc = nlp(text)
    
    # Display dependency tree
    spacy.displacy.render(doc, style="dep", jupyter=True)

def visualize_literal_tree(text, pos_tags=True):
    """
    Visualize the dependency tree as a literal tree structure using pygraphviz.
    If pos_tags is True, display POS tags instead of tokens in the tree.
    
    Args:
    - text (str): The text to visualize the dependency tree for.
    - pos_tags (bool): Whether to display POS tags instead of the token text (default is False).
    """
    doc = nlp(text)
    
    # Create a new directed graph
    G = pgv.AGraph(strict=False, directed=True)

    # Add nodes and edges based on the dependency parsing
    for token in doc:
        # If pos_tags is True, display POS tags; otherwise, display the token text
        label = f"{token.text}\n({token.pos_})" if pos_tags else token.text
        G.add_node(token.i, label=label)
        if token.head != token:
            G.add_edge(token.head.i, token.i)

    # Render
    G.layout(prog="dot")


def feature_engineering(df):
    """create new features based on the headline text"""
    # list of functional words (POS tags)
    func_pos_tags = ['ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PUNCT', 'SCONJ']
    
    # 7 most common POS tags
    common_pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'PROPN']

    ratio_func_words = []
    ratio_unique_pos = []
    lengths = []
    pos_repetitions = []
    
    pos_tag_ratios = {pos_tag: [] for pos_tag in common_pos_tags}

    for _, row in df.iterrows():
        headline = row['headline']
        doc = nlp(headline)
        
        total_words = len(doc)
        
        # Calculate ratio_func_words
        func_words = sum(1 for token in doc if token.pos_ in func_pos_tags)
        ratio_func_words.append(func_words / total_words if total_words > 0 else 0)

        # Calculate ratio_unique_pos
        unique_pos = len(set(token.pos_ for token in doc))
        ratio_unique_pos.append(unique_pos / total_words if total_words > 0 else 0)

        # Calculate length
        lengths.append(total_words)

        # check for consecutive POS tag repetitions
        pos_repetition_found = any(doc[i].pos_ == doc[i+1].pos_ for i in range(len(doc)-1))
        pos_repetitions.append(1 if pos_repetition_found else 0)
        
        # Calculate ratios for 7 most common POS tags
        for pos_tag in common_pos_tags:
            tag_count = sum(1 for token in doc if token.pos_ == pos_tag)
            pos_tag_ratios[pos_tag].append(tag_count / total_words if total_words > 0 else 0)

    df['ratio_func_words'] = ratio_func_words
    df['ratio_unique_pos'] = ratio_unique_pos
    df['length'] = lengths
    df['pos_repetitions'] = pos_repetitions

    for pos_tag in common_pos_tags:
        df[f'ratio_{pos_tag.lower()}'] = pos_tag_ratios[pos_tag]

    return df


def calculate_syntactic_complexity(doc):
    """
    Calculate the syntactic complexity of a sentence based on its dependency tree.
    Syntactic complexity is defined as a combination of tree depth and average branching factor.
    This function scales the complexity score from 0 to 1.
    """
    
    # depth of the syntactic tree
    def get_tree_depth(token):
        if not any(child for child in token.children):  # If the token has no children, it's a leaf
            return 1
        return 1 + max(get_tree_depth(child) for child in token.children)
    
    # Get root token (typically main verb or root of  sentence)
    root_token = [token for token in doc if token.dep_ == 'ROOT'][0]
    
    tree_depth = get_tree_depth(root_token)
    
    # Calculate the average branching factor
    total_children = sum(len([child for child in token.children]) for token in doc)
    num_non_leaf_nodes = sum(1 for token in doc if len([child for child in token.children]) > 0)
    average_branching_factor = total_children / num_non_leaf_nodes if num_non_leaf_nodes > 0 else 0
    
    return tree_depth, average_branching_factor


def add_syntactic_features(df):
    """add syntactic features to the dataframe"""
    depths = []
    branching_factors = []

    for _, row in df.iterrows():
        headline = row['headline']
        doc = nlp(headline) 
        
        depth, branching_factor = calculate_syntactic_complexity(doc)        
        depths.append(depth)
        branching_factors.append(branching_factor)

    df['syntactic_depth'] = depths
    df['branching_factor'] = branching_factors
    

    return df

# Load GPT-2 Model and Tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def get_word_probabilities(sentence):
    """Function to calculate the probability of each word in a sentence using GPT-2 model."""
    tokens = tokenizer.encode(sentence, return_tensors="pt")
    
    # Get the model's output (logits) for the sentence
    with torch.no_grad():
        outputs = model(tokens)
    logits = outputs.logits.squeeze()
    
    # Calculate the probability for each word in the sentence
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Decode the tokens back to words
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokens[0])
    
    word_probabilities = {}
    
    # For each word, find the probability
    for i, token in enumerate(decoded_tokens[1:]):  # Start from 1 to skip the beginning token
        word_probabilities[token] = probabilities[i, tokenizer.convert_tokens_to_ids(token)].item()
    
    return word_probabilities


def calculate_surprisingness_features(df):
    """Calculate the reversed probability (most surprising word) and average surprisingness for each headline."""
    reversed_probabilities = []
    average_surprisingness = []
    
    for _, row in df.iterrows():
        headline = row['headline']
        
        word_probabilities = get_word_probabilities(headline)
        
        if word_probabilities:
            most_surprising_word = max(word_probabilities, key=word_probabilities.get)
            most_surprising_probability = word_probabilities[most_surprising_word]
            reversed_probability = 1 - most_surprising_probability
            
            avg_surprisingness = sum([1 - prob for prob in word_probabilities.values()]) / len(word_probabilities)
        else:
            reversed_probability = 0
            avg_surprisingness = 0
        
        reversed_probabilities.append(reversed_probability)
        average_surprisingness.append(avg_surprisingness)
    
    df['reversed_probability'] = reversed_probabilities
    df['average_surprisingness'] = average_surprisingness

    return df


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python syntactic_feature_eng.py <input_json_path> <output_csv_path>")
        sys.exit(1)

    input_path = os.path.normpath(sys.argv[1])
    output_path = os.path.normpath(sys.argv[2])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_json(input_path)

    df = feature_engineering(df)
    df = add_syntactic_features(df)
    df = calculate_surprisingness_features(df)

    df.to_csv(output_path, index=False)

    print(f"DataFrame saved as CSV at: {output_path}")