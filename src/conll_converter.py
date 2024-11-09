import os
import spacy
import pandas as pd


def convert_to_conllu(df, output_file, model):
    """
    Converts a DataFrame to CoNLL-U format.
    
    Args:
        df (pd.DataFrame): DataFrame with 'is_sarcastic', 'headline', and 'article_link' columns.
        output_file (str): File path for saving the CoNLL-U formatted data.
    """
    conllu_data = []

    for index, row in df.iterrows():
        # Extract row data
        sent_id = index + 1
        label = row["is_sarcastic"]
        sentence = row["headline"]
        link = row["article_link"]

        # Process sentence using spaCy
        doc = model(sentence)

        # Add sentence-level metadata
        sentence_lines = [
            f"# sent_id = {sent_id}",
            f"# class = {label}",
            f"# link = {link}"
        ]

        # Add token-level annotations
        for i, token in enumerate(doc):
            sentence_lines.append(
                f"{i+1}\t{token.text}\t{token.lemma_}\t{token.pos_}\t_\t_\t0\t_\t_\t_"
            )

        # Add the processed sentence to the dataset
        conllu_data.append("\n".join(sentence_lines))

    # Write the CoNLL-U data to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(conllu_data) + "\n")


if __name__ == "__main__":
    output_folder = "../Data/"
    output_file = os.path.join("../Data", "dataset.conllu")
    nlp = spacy.load("en_core_web_sm")
    file_path = "../Data/Sarcasm_Headlines_Dataset.json"
    data = pd.read_json(file_path, lines=True)
    convert_to_conllu(data, output_file, nlp)