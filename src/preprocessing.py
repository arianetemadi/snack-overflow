import os
import spacy
import pandas as pd
from tqdm import tqdm


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
                f"# link = {link}"
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


if __name__ == "__main__":
    output_file = os.path.join("../data", "dataset.conllu")
    nlp = spacy.load("en_core_web_sm")
    file_path = "../data/Sarcasm_Headlines_Dataset.json"
    data = pd.read_json(file_path, lines=True)
    convert_to_conllu(data, output_file, nlp)