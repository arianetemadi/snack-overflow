# imports
from sklearn.model_selection import train_test_split as split
from src.data_util import load_data
from src.naive_bayes import NaiveBayesClassifier
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import random


class LRClassifier:
    def __init__(self, train_data, val_data, use_tfidf=False, remove_stopwords=False):
        self.train_data = train_data
        self.val_data = val_data
        self.use_tfidf = use_tfidf
        self.remove_stopwords = remove_stopwords
        stop_words = 'english' if remove_stopwords else None
        self.vectorizer = TfidfVectorizer(stop_words=stop_words) if use_tfidf else CountVectorizer(stop_words=stop_words)
        self.model = LogisticRegression(random_state=42)

    def preprocess(self, data):
        """Extracts headlines and their labels from the data."""
        headlines = []
        labels = []
        for headline_group in data:
            full_headline = " ".join([sentence.metadata['text'] for sentence in headline_group])
            headlines.append(full_headline)
            labels.append(int(headline_group[0].metadata['class'])) 
        return headlines, labels

    def fit(self):
        """Fits the logistic regression model on the training data."""
        train_headlines, train_labels = self.preprocess(self.train_data)
        train_features = self.vectorizer.fit_transform(train_headlines)
        self.model.fit(train_features, train_labels)

    def predict(self):
        """Predicts the labels for the validation data."""
        # preprocess validation data
        val_headlines, val_labels = self.preprocess(self.val_data)

        # vectorize validation headlines
        val_features = self.vectorizer.transform(val_headlines)

        # predict labels and probabilities
        predictions = self.model.predict(val_features)
        probabilities = self.model.predict_proba(val_features)[:, 1] 
        return val_headlines, val_labels, predictions, probabilities

    def evaluate(self, show_confusion_matrix=False):
        """Evaluates the model and stores false positives and negatives."""
        val_headlines, val_labels, predictions, probabilities = self.predict()

        # classification report
        print("Classification Report:")
        print(classification_report(val_labels, predictions))

        if show_confusion_matrix:
            cm = confusion_matrix(val_labels, predictions)
            random_cmap = random.choice(["Blues", "Reds", "Greens", "YlOrBr"])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap=random_cmap,
                        xticklabels=["Not Sarcastic", "Sarcastic"],
                        yticklabels=["Not Sarcastic", "Sarcastic"])
            plt.title("Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.show()

        # store false positives and false negatives
        val_labels = np.array(val_labels)
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        false_positives = [
            {
                "Index": i,
                "Headline": val_headlines[i],
                "True Class": val_labels[i],
                "Predicted Class": predictions[i],
                "Probability": probabilities[i],
            }
            for i in range(len(val_labels)) if predictions[i] == 1 and val_labels[i] == 0
        ]

        false_negatives = [
            {
                "Index": i,
                "Headline": val_headlines[i],
                "True Class": val_labels[i],
                "Predicted Class": predictions[i],
                "Probability": probabilities[i],
            }
            for i in range(len(val_labels)) if predictions[i] == 0 and val_labels[i] == 1
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
        # get feature names (words)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # get the coefficients
        coefficients = self.model.coef_.flatten()
        top_positive_idx = coefficients.argsort()[-n_top_features:][::-1] 
        top_negative_idx = coefficients.argsort()[:n_top_features] 

        #positive and negative coefficients
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

        # plot
        plt.figure(figsize=(10, 6))
        plt.barh(top_positive_words, top_positive_coeffs, color='green', alpha=0.7, label='Positive (Sarcastic)')
        plt.barh(top_negative_words, top_negative_coeffs, color='red', alpha=0.7, label='Negative (Non-sarcastic)')
        plt.xlabel('Coefficient Value')
        plt.title('Top Features Learned by Logistic Regression')
        plt.legend()
        plt.gca().invert_yaxis() 
        plt.show()
