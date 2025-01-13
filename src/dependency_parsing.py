import json
import spacy
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

input_file = "data\Sarcasm_Headlines_Dataset.json"
output_file = "data\headlines_depency_parsed.json"

def parse_data(input_file, output_file):
    parsed_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            headline = data["headline"]

            doc = nlp(headline)

            pos_tags = [{"text": token.text, "pos": token.pos_, "dep": token.dep_, "head": token.head.text} for token in doc]
            syntax_tree = [{"text": token.text, "dep": token.dep_, "children": [child.text for child in token.children]} for token in doc]

            parsed_data.append({
                "is_sarcastic": data["is_sarcastic"],
                "headline": headline,
                "article_link": data["article_link"],
                "pos_tags": pos_tags,
                "syntax_tree": syntax_tree
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4, ensure_ascii=False)

    print(f"Data has been parsed and saved to {output_file}")

# Run parser
parse_data(input_file, output_file)
