import json
import spacy
import argparse
import sys

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

def parse_data(input_file, output_file):
    parsed_data = []

    #complicated because of the json line format
    with open(input_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:  
                print(f"Skipping empty line at {line_number}")
                continue

            try:
                data = json.loads(line)  # Parse JSON
                headline = data.get("headline", "")

                # Process with spaCy
                doc = nlp(headline)

                # Extract POS tags syntax tree
                pos_tags = [{"text": token.text, "pos": token.pos_, "dep": token.dep_, "head": token.head.text} for token in doc]
                syntax_tree = [{"text": token.text, "dep": token.dep_, "children": [child.text for child in token.children]} for token in doc]

                # Append
                parsed_data.append({
                    "is_sarcastic": data.get("is_sarcastic", None),
                    "headline": headline,
                    "article_link": data.get("article_link", ""),
                    "pos_tags": pos_tags,
                    "syntax_tree": syntax_tree
                })
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON at line {line_number}: {e}")
                continue

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_data, f, indent=4, ensure_ascii=False)

    print(f"Data has been parsed and saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_headlines.py <input_file> <output_file>")
        print("Example: python parse_headlines.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    parse_data(input_file, output_file)
