import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn.metrics import classification_report


class NaiveBayesClassifier:
    def __init__(self, ngram_range=(1, 1)):
        self.model = MultinomialNB()
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            preprocessor=lambda x: x,
            tokenizer=lambda x: x,
        )

    def fit(self, docs):
        # create a corpus for the CountVectorizer
        corpus = []
        for doc in docs:
            for sentence in doc:
                words = self.extract_words(sentence)
                corpus.append(words)

        # fit the CountVectorizer on corpus
        self.vectorizer.fit(corpus)

        # fit the model on input docs
        classes = [0, 1]
        for doc in tqdm(docs):
            # create feature vectors
            words = self.extract_words(doc[0])
            X = self.vectorizer.transform([words])
            for sentence in doc[1:]:
                words = self.extract_words(sentence)
                X += self.vectorizer.transform([words])
            y = np.array([int(doc[0].metadata["class"])])

            # add the features in this doc to the fit
            self.model.partial_fit(X.reshape(1, -1), y, classes)

    def extract_words(self, sentence):
        return [token["lemma"] for token in sentence if token["upos"] != "PUNCT"]

    def test(self, docs):
        y_true = []
        y_pred = []
        fp = []
        fn = []
        for doc in tqdm(docs):
            words = self.extract_words(doc[0])
            X = self.vectorizer.transform([words])
            for sentence in doc[1:]:
                words = self.extract_words(sentence)
                X += self.vectorizer.transform([words])
            y = np.array([int(doc[0].metadata["class"])])
            y_pred.append(self.model.predict(X)[0])
            y_true.append(y[0])

            # collect false positive and false negatives
            if y_pred[-1] != y_true[-1]:
                if y_true[-1] == 0:
                    fp.append(doc)
                else:
                    fn.append(doc)

        target_names = ["Non-sarcastic", "Sarcastic"]
        print(classification_report(y_true, y_pred, target_names=target_names))

        return fp, fn

    def show_word_weights(self, doc):
        print(("{:>14}" * 4).format("word", "sarcastic", "non-sarcastic", "diff"))
        print("=" * 56)
        threshold = 1
        for sentence in doc:
            for token in sentence:
                if token["lemma"] in self.vectorizer.vocabulary_:
                    neg_weight = self.model.feature_log_prob_[0][
                        self.vectorizer.transform([[token["lemma"]]]).nonzero()[1][0]
                    ]
                    pos_weight = self.model.feature_log_prob_[1][
                        self.vectorizer.transform([[token["lemma"]]]).nonzero()[1][0]
                    ]
                    diff = pos_weight - neg_weight
                else:
                    pos_weight, neg_weight, diff = -1, -1, 0
                p_token = (
                    token["form"] if abs(diff) < threshold else f"*{token['form']}"
                )
                print(
                    f"{p_token:>14}{pos_weight:>14.2f}{neg_weight:>14.2f}{diff:>14.2f}"
                )
        print()

    def show_decisive_words(self, n=10):
        ret = []
        for word in self.vectorizer.vocabulary_:
            neg_weight = self.model.feature_log_prob_[0][
                self.vectorizer.transform([[word]]).nonzero()[1][0]
            ]
            pos_weight = self.model.feature_log_prob_[1][
                self.vectorizer.transform([[word]]).nonzero()[1][0]
            ]
            diff = pos_weight - neg_weight
            ret.append((diff, word))
        ret = sorted(ret)
        return ret[:n], ret[-n:]
