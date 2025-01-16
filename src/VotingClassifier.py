import pandas as pd
import conllu
from src.data_util import load_data
from src.naive_bayes import NaiveBayesClassifier
from src.LogisticRegression import LRClassifier
from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report


class VotingClassifier:
    def __init__(self, conllu_path=r"C:\Users\MSC\OneDrive - Fraunhofer Austria Research GmbH\Desktop\NLP\data\dataset.conllu",
                  csv_path=r"C:\Users\MSC\OneDrive - Fraunhofer Austria Research GmbH\Desktop\NLP\data\trans_prob_temp.csv",
                    other_data = None):

    
        self.conllu_path = conllu_path
        self.csv_path = csv_path
        self.new_data = False
        self.other_data_path = other_data
        headlines = load_data(conllu_path)
        data = pd.read_csv(csv_path)
        data = data.drop(columns=['pos_tags', 'syntax_tree'])

        self.y = data['is_sarcastic']
        self.X_syn = data.drop(columns=['headline', 'is_sarcastic'])

        # split into training and test sets
        SEED = 42
        self.train_headlines, other_headlines = split(headlines, test_size=0.3, random_state=SEED)
        self.val_headlines, self.test_headlines = split(other_headlines, test_size=0.5, random_state=SEED)
        self.train_data, other_data = split(data, test_size=0.3, random_state=SEED)
        self.val_data, self.test_data = split(other_data, test_size=0.5, random_state=SEED)

        # if other_data is not None:
        #     self.other_conllu = other_data[0]
        #     self.other_csv = other_data[1]
        #     self.val_headlines = load_data(self.other_conllu)
        #     temp = data = pd.read_csv(csv_path)
        #     temp = temp.drop(columns=['pos_tags', 'syntax_tree'])
        #     self.y_val = temp['is_sarcastic']
        #     self.X_syn_val = temp.drop(columns=['headline', 'is_sarcastic'])
        #     self.val_data = self.X_syn_val

        

    def fit_modles(self):
        ############################## Naive Bayes ##############################

        # fit the Naive Bayes Bag of Word model to training data
        self.naive_bayes = NaiveBayesClassifier(ngram_range=(1, 1))
        self.naive_bayes.fit(self.train_headlines)
        # nb_predictions = naive_bayes.predict(self.val_headlines)
        # nb_probabilities = naive_bayes.predict_proba(self.val_headlines)

        # ############################## Logistic Regression ##############################

        # # fit the Logistic Regression model to training data
        # self.logistic_regression = LRClassifier(use_tfidf=False, remove_stopwords=False)
        # self.logistic_regression.fit(self.train_headlines)
        # # lr_predictions = logistic_regression.predict(self.val_headlines)
        # # lr_probabilities = logistic_regression.predict_proba(self.val_headlines)

        ############################## Syntactic Logistic regression ##############################

        self.top_features_lr = ['ratio_func_words', 'ratio_unique_pos', 'ratio_verb', 'ratio_adv',
       'ratio_pron']
        
        X_syn_lr = self.train_data[self.top_features_lr]
        y_train = self.train_data['is_sarcastic']
        
        ## Logistic Regression
        # Train logistic regression classifier on X_syn_lr
        self.logistic_regression_syn_lr = LogisticRegression(max_iter=1000)
        self.logistic_regression_syn_lr.fit(X_syn_lr, y_train)

        ############################## Syntactic Random Forest ##############################

        self.top_features_rf = ['average_surprisingness', 'reversed_probability', 'ratio_func_words',
       'ratio_verb', 'ratio_unique_pos']
        
        X_syn_rf = self.train_data[self.top_features_rf]
        
        ## Random Forest    
        # Train random forest classifier on X_syn_rf
        self.random_forest_syn_rf = RandomForestClassifier()
        self.random_forest_syn_rf.fit(X_syn_rf, y_train)


    def predict(self, new_data=None):
        if new_data is not None:

            self.new_data=True
            self.val_headlines = load_data(new_data[0])
            temp = pd.read_csv(new_data[1])
            temp = temp.drop(columns=['pos_tags', 'syntax_tree'])
            self.y_val = temp['is_sarcastic']
            self.X_syn_val = temp.drop(columns=['headline', 'is_sarcastic'])
            self.val_data = self.X_syn_val

            self.predictions = {
                'naive_bayes': self.naive_bayes.predict(self.val_headlines),
                'logistic_regression_syn_lr': self.logistic_regression_syn_lr.predict(self.val_data[self.top_features_lr]),
                'random_forest_syn_rf': self.random_forest_syn_rf.predict(self.val_data[self.top_features_rf])
            }
            return self.predictions
        else:
            self.predictions = {
                'naive_bayes': self.naive_bayes.predict(self.val_headlines),
                'logistic_regression_syn_lr': self.logistic_regression_syn_lr.predict(self.val_data[self.top_features_lr]),
                'random_forest_syn_rf': self.random_forest_syn_rf.predict(self.val_data[self.top_features_rf])
            }
            return self.predictions

    def evaluate(self):
        models = {
            'naive_bayes': self.naive_bayes,
            'logistic_regression_syn_lr': self.logistic_regression_syn_lr,
            'random_forest_syn_rf': self.random_forest_syn_rf
        }
        if self.new_data:
            for model in models:
                print(f"Evaluating {model}")
                print(classification_report(self.y_val, self.predictions[model]))
                print("Confusion Matrix:")
                print(confusion_matrix(self.y_val, self.predictions[model])) 

        else:
            for model in models:
                print(f"Evaluating {model}")
                print(classification_report(self.val_data['is_sarcastic'], self.predictions[model]))
                print("Confusion Matrix:")
                print(confusion_matrix(self.val_data['is_sarcastic'], self.predictions[model]))

    def voting(self):
        self.voting_predictions = []
        for i in range(len(self.val_data)):
            vote_sum = sum(self.predictions[model][i] for model in self.predictions)
            if vote_sum <= 1:
                self.voting_predictions.append(0)
            else:
                self.voting_predictions.append(1)
        return self.voting_predictions
    
    def eval_voting(self):
        if self.new_data:
            print("Evaluating Voting Classifier")
            print(classification_report(self.y_val, self.voting_predictions))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_val, self.voting_predictions))
        else:
            print("Evaluating Voting Classifier")
            print(classification_report(self.val_data['is_sarcastic'], self.voting_predictions))
            print("Confusion Matrix:")
            print(confusion_matrix(self.val_data['is_sarcastic'], self.voting_predictions))

    