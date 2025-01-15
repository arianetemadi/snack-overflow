"""
This approach uses the POS tags extracted during the preprocessing
to dig deeper and discover whether there are any patterns to be
uncoverred from the structure of the headlines only. Unfortunately, 
this approach yields very poor results.
"""
from gensim.models import Word2Vec
from conllu import parse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.data_util import load_data


def extract_pos_tags(token_list):
    return [token['upos'] for token in token_list]


def encode_sequence(sequence, model):
    return np.mean([model.wv[tag] for tag in sequence if tag in model.wv], axis=0)


def fit_and_predict(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=123)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))


def main():
    with open("../data/dataset.conllu") as f:
       lines = f.readlines()
    sentences = parse("".join(lines))
    pos_tags = list(map(extract_pos_tags, sentences))
    labels = [int(sentence.metadata['class']) for sentence in sentences]

    model = Word2Vec(min_count=1, vector_size=100, window=5, sg=1)
    model.build_vocab(pos_tags)
    model.train(pos_tags, total_examples=model.corpus_count, epochs=model.epochs)
    feature_vectors = np.array([encode_sequence(seq, model) for seq in pos_tags])
    fit_and_predict(feature_vectors, labels)

if __name__ == "__main__":
    main()