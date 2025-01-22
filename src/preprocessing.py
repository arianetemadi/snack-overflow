import os
import spacy
import pandas as pd
from tqdm import tqdm
import sys
import re
import json


def convert_to_conllu(df, output_file, model):
    """
    Converts a DataFrame to CoNLL-U format.

    Args:
        df (pd.DataFrame): DataFrame with 'is_sarcastic', 'headline', and 'article_link' columns.
        output_file (str): File path for saving the CoNLL-U formatted data.
    """
    conllu_data = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Extract row data
        headline_id = index + 1
        label = row["is_sarcastic"]
        headline = row["headline"]
        link = row["article_link"]

        # Process headline using spaCy
        doc = model(headline)

        # Process sentences (headlines can have more than one sentence)
        for i, sentence in enumerate(doc.sents):

            # Add metadata
            sentence_lines = [
                f"# text = {sentence}",
                f"# headline_id = {headline_id}",
                f"# sent_id = {i}",
                f"# class = {label}",
                f"# link = {link}",
            ]

            # Add token-level annotations
            for j, token in enumerate(sentence):
                sentence_lines.append(
                    f"{j+1}\t{token.text}\t{token.lemma_}\t{token.pos_}\t_\t_\t{token.head.i}\t_\t_\t_"
                )

            # Add the processed sentence to the dataset
            conllu_data.append("\n".join(sentence_lines))

    # Write the CoNLL-U data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(conllu_data) + "\n")


def convert_txt_to_json(input_file, output_file, link):
    """
    Converts headlines in input_file to JSON and writes to output_file.

    Args:
        input_file (str): File path of the txt file including the input headlines.
            Lines containing "0" or "1" specify the label for the following headlines
            (until another label arrives).
            Empty lines are ignored.
        output_file (str): File path for saving the converted JSON data.
        link (str): Link to the source of the headlines.
    """

    # read lines from the input text file
    headlines = []
    labels = []
    links = []
    current_label = "0"
    with open(input_file) as file:
        for line in file:
            line = line.strip().strip('"”“.').strip().lower()
            if len(line) == 0:
                continue
            if len(line) == 1:
                current_label = line
                continue
            headlines.append(line)
            labels.append(current_label)
            links.append(link)

    # convert to pandas dataframe
    df = pd.DataFrame(
        {"is_sarcastic": labels, "headline": headlines, "article_link": links}
    )

    # shuffle rows randomly
    df = df.sample(frac=1, random_state=1234).reset_index(drop=True)

    # write to file
    df.to_json(output_file, orient="records", lines=True, force_ascii=True)


def preprocess_tweets(csv_path, out_path):
    """
    Convert the dataset of tweets into json format
    comparible with the headlines dataset. Sanitize
    the tweets by removing user names (@-tags) and
    unnecessary whitespaces.

    Args:
        csv_path(str): location of the source csv file with tweets
        out_path(str): where to store the output json
    """
    data = pd.read_csv(csv_path)
    tweets = [re.sub(r"@\w+", "", str(tweet).replace("\n", "")) for tweet in data["tweet"]]
    labels = data["sarcastic"]
    
    with open(out_path, "w") as f:
        for label, tweet in zip(labels, tweets):
            f.write(json.dumps({"is_sarcastic": label, "headline": tweet, "article_link": ""}) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python preprocessing.py <input_filename> <output_filename>")
        sys.exit(1)

    #output_file = os.path.join("..\data", sys.argv[2])
    output_file = sys.argv[2]
    nlp = spacy.load("en_core_web_sm")
    #file_path = os.path.join("..\data", sys.argv[1])
    file_path = sys.argv[1]
    data = pd.read_json(file_path, lines=True)
    convert_to_conllu(data, output_file, nlp)
