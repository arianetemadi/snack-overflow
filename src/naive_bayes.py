import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
from sklearn.metrics import classification_report


class NaiveBayesClassifier:
    def __init__(self, ngram_range=(1, 1)):
        self.model = MultinomialNB()
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(ngram_range=self.ngram_range, preprocessor=lambda x: x, tokenizer=lambda x: x)

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
            
        target_names = ['Non-sarcastic', 'Sarcastic']
        print(classification_report(y_true, y_pred, target_names=target_names))

        return fp, fn