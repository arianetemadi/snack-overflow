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
            line = line.strip().lower()
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


if __name__ == "__main__":
    output_file = os.path.join("../data", "dataset.conllu")
    nlp = spacy.load("en_core_web_sm")
    file_path = "../data/Sarcasm_Headlines_Dataset.json"
    data = pd.read_json(file_path, lines=True)
    convert_to_conllu(data, output_file, nlp)
