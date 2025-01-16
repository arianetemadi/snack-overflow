from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm


class LRClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, use_tfidf=False, remove_stopwords=False):
        self.use_tfidf = use_tfidf
        self.remove_stopwords = remove_stopwords
        stop_words = 'english' if remove_stopwords else None
        self.vectorizer = TfidfVectorizer(stop_words=stop_words,
                                          ngram_range=(1, 1),
                                          preprocessor=lambda x: x,
                                          tokenizer=lambda x: x) if use_tfidf else CountVectorizer(ngram_range=(1, 1),
                                                                                                  preprocessor=lambda x: x,
                                                                                                  tokenizer=lambda x: x,)
        self.model = SGDClassifier(loss='log_loss', random_state=42)

    def preprocess(self, data):
        """Extracts headlines and their labels from the data."""
        headlines = []
        labels = []
        for headline_group in data:
            full_headline = " ".join([sentence.metadata['text'] for sentence in headline_group])
            headlines.append(full_headline)
            labels.append(int(headline_group[0].metadata['class']))
        return headlines, labels

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
            self.X = self.vectorizer.transform([words])
            for sentence in doc[1:]:
                words = self.extract_words(sentence)
                self.X += self.vectorizer.transform([words])
            self.y = np.array([int(doc[0].metadata["class"])])

            self.model.partial_fit(self.X.reshape(1, -1), self.y, classes)
        return self
    
    def extract_words(self, sentence):
        return [token["lemma"] for token in sentence if token["upos"] != "PUNCT"]

    # def predict(self):
    #     """Predicts the labels for the given data."""
    #     # Vectorize the input data
    #     X_features = self.vectorizer.transform(self.X)
    #     return self.model.predict(X_features)

    # def predict_proba(self):
    #     """Predicts probabilities for the given data."""
    #     X_features = self.vectorizer.transform(self.X)
    #     return self.model.predict_proba(X_features)
    
    def predict(self, docs):
        y_pred = []
        y_true = []
        for doc in tqdm(docs):
            words = self.extract_words(doc[0])
            X = self.vectorizer.transform([words])
            for sentence in doc[1:]:
                words = self.extract_words(sentence)
                X += self.vectorizer.transform([words])
            y_pred.append(self.model.predict(X)[0])

        return y_pred

    def predict_proba(self, docs):
        y_prob = []
        for doc in tqdm(docs):
            words = self.extract_words(doc[0])
            X = self.vectorizer.transform([words])
            for sentence in doc[1:]:
                words = self.extract_words(sentence)
                X += self.vectorizer.transform([words])
            y_prob.append(self.model.predict_proba(X)[0])
            

        return y_prob

    def score(self):
        """Calculates the accuracy score on the given data."""
        predictions = self.predict(self.X)
        return accuracy_score(self.y, predictions)

    def evaluate(self,docs, show_confusion_matrix=False):
        """Evaluates the model and stores false positives and negatives."""

        # fit the model on input docs
        headlines = []
        y_val_truth = []
        for doc in tqdm(docs):
            # create feature vectors
            words = self.extract_words(doc[0])
            headlines.append(" ".join(words))
            y_val_truth.append(int(doc[0].metadata["class"]))
        # Predictions and probabilities
        predictions = self.predict(docs)
        probabilities = self.predict_proba(docs)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_val_truth, predictions))

        if show_confusion_matrix:
            cm = confusion_matrix(y_val_truth, predictions)
            random_cmap = random.choice(["Blues", "Reds", "Greens", "YlOrBr"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap=random_cmap,
                        xticklabels=["Not Sarcastic", "Sarcastic"],
                        yticklabels=["Not Sarcastic", "Sarcastic"])
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.show()

        # Store false positives and false negatives
        self.y = np.array(y_val_truth)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        false_positives = [
            {
                "Index": i,
                "Headline": headlines[i],
                "True Class": y_val_truth[i],
                "Predicted Class": predictions[i],
                "Probability": probabilities[i],
            }
            for i in range(len(self.y)) if predictions[i] == 1 and y_val_truth[i] == 0
        ]

        false_negatives = [
            {
                "Index": i,
                "Headline": headlines[i],
                "True Class": y_val_truth[i],
                "Predicted Class": predictions[i],
                "Probability": probabilities[i],
            }
            for i in range(len(y_val_truth)) if predictions[i] == 0 and y_val_truth[i] == 1
        ]

        self.false_positives_df = pd.DataFrame(false_positives)
        self.false_negatives_df = pd.DataFrame(false_negatives)

        print("False positives and false negatives stored in self.false_positives_df and self.false_negatives_df.")

    def interpret_model(self, n_top_features=20):
        """
        Interprets the Logistic Regression model by displaying the most important features (words)
        and their corresponding coefficients in a DataFrame.

        Args:
        - n_top_features: Number of top features to display (default is 20).
        """
        # Get feature names (words)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # Get the coefficients
        coefficients = self.model.coef_.flatten()
        top_positive_idx = coefficients.argsort()[-n_top_features:][::-1]
        top_negative_idx = coefficients.argsort()[:n_top_features]

        # Positive and negative coefficients
        top_positive_words = feature_names[top_positive_idx]
        top_positive_coeffs = coefficients[top_positive_idx]
        top_negative_words = feature_names[top_negative_idx]
        top_negative_coeffs = coefficients[top_negative_idx]

        # DataFrame for top words and coefficients
        top_features = pd.DataFrame({
            'Word': np.concatenate([top_positive_words, top_negative_words]),
            'Coefficient': np.concatenate([top_positive_coeffs, top_negative_coeffs]),
            'Sentiment': ['Sarcastic'] * n_top_features + ['Non-Sarcastic'] * n_top_features
        })
        display(top_features)

        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(top_positive_words, top_positive_coeffs, color='green', alpha=0.7, label='Positive (Sarcastic)')
        plt.barh(top_negative_words, top_negative_coeffs, color='red', alpha=0.7, label='Negative (Non-sarcastic)')
        plt.xlabel('Coefficient Value')
        plt.title('Top Features Learned by Logistic Regression')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.show()
