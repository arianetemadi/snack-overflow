# Simple BoW NN using the implementation from the lecture
# Source: https://github.com/tuw-nlp-ie/tuw-nlp-ie-2024WS/blob/main/lectures/05_Deep_learning_practical_lesson/deep_learning_practical_lesson_without_outputs.ipynb

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim
from sklearn.model_selection import train_test_split as split
from src.data_util import load_data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


SEED = 42


def create_dataloader_iterator(X, y, word_to_ix, shuffle, batch_size=64):
    vecs = torch.FloatTensor(word_to_ix.transform(X).toarray())
    labels = torch.LongTensor(y)
    data_loader = [(sample, label) for sample, label in zip(vecs, labels)]
    return DataLoader(
        data_loader,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def calculate_performance(preds, y):
    """
    Returns precision, recall, fscore per batch
    """
    rounded_preds = preds.argmax(1)

    precision, recall, fscore, support = precision_recall_fscore_support(
        rounded_preds.cpu(), y.cpu(), zero_division=np.nan
    )

    return precision[0], recall[0], fscore[0]


class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec, _):
        return F.log_softmax(self.linear(bow_vec), dim=1)


class BoWNN:
    def __init__(self, output_dim, input_dim):
        self.model = BoWClassifier(output_dim, input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.NLLLoss()

    def predict(self, iterator):
        self.model.eval()

        preds = []

        with torch.no_grad():

            for batch in iterator:
                text_vecs = batch[0]
                labels = batch[1]
                sen_lens = []

                if len(batch) > 2:
                    sen_lens = batch[2]

                predictions = self.model(text_vecs, sen_lens)
                preds.extend([t.item() for t in predictions.argmax(1)])

            return preds

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0
        self.model.eval()

        preds = []

        with torch.no_grad():

            for batch in iterator:
                text_vecs = batch[0]
                labels = batch[1]
                sen_lens = []

                if len(batch) > 2:
                    sen_lens = batch[2]

                predictions = self.model(text_vecs, sen_lens)
                loss = self.criterion(predictions, labels)

                prec, recall, fscore = calculate_performance(predictions, labels)

                epoch_loss += loss.item()
                epoch_prec += prec.item()
                epoch_recall += recall.item()
                epoch_fscore += fscore.item()

                preds.extend(predictions)

        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    def train(self, iterator):
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        self.model.train()

        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]
            sen_lens = []

            if len(batch) > 2:
                sen_lens = batch[2]

            self.optimizer.zero_grad()

            predictions = self.model(text_vecs, sen_lens)

            loss = self.criterion(predictions, labels)

            prec, recall, fscore = calculate_performance(predictions, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_prec += prec.item()
            epoch_recall += recall.item()
            epoch_fscore += fscore.item()
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    def training_loop(self, train_iterator, valid_iterator, epoch_number):
        # early stopping
        prev_loss = None
        curr_loss = np.inf

        # max number of epochs
        epoch = 0

        while epoch < epoch_number and (prev_loss is None or curr_loss < prev_loss):
            train_loss, train_prec, train_rec, train_fscore = self.train(train_iterator)

            valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(
                valid_iterator
            )

            prev_loss = curr_loss
            curr_loss = valid_loss
            epoch += 1

            print(
                f"\t[{epoch:2d}] Train Loss: {train_loss:.3f} | Train Prec: {train_prec*100:.2f}% | Train Rec: {train_rec*100:.2f}% | Train Fscore: {train_fscore*100:.2f}%"
            )
            print(
                f"\t[{epoch:2d}]   Val Loss: {valid_loss:.3f} |   Val Prec: {valid_prec*100:.2f}% |   Val Rec: {valid_rec*100:.2f}% |   Val Fscore: {valid_fscore*100:.2f}%"
            )


def main(data_path, n_epochs=50):
    data = load_data(data_path)
    headlines = list(
        map(lambda line: "".join(list(map(lambda x: x.metadata["text"], line))), data)
    )
    labels = list(map(lambda line: int(line[0].metadata["class"]), data))

    X_train, other_data = split(headlines, test_size=0.3, random_state=SEED)
    X_val, X_test = split(other_data, test_size=0.5, random_state=SEED)

    y_train, other_data = split(labels, test_size=0.3, random_state=SEED)
    y_val, y_test = split(other_data, test_size=0.5, random_state=SEED)

    vectorizer = CountVectorizer(max_features=6000)

    word_to_ix = vectorizer.fit(X_train)

    train_iterator = create_dataloader_iterator(X_train, y_train, word_to_ix, True)
    valid_iterator = create_dataloader_iterator(X_val, y_val, word_to_ix, False)
    test_iterator = create_dataloader_iterator(X_test, y_test, word_to_ix, False)

    bow_nn = BoWNN(2, len(word_to_ix.vocabulary_))

    bow_nn.training_loop(train_iterator, valid_iterator, n_epochs)

    loss, prec, rec, fscore = bow_nn.evaluate(test_iterator)
    print(
        f"\t Test Loss: {loss:.3f} |  Test Prec: {prec*100:.2f}% |  Test Rec: {rec*100:.2f}% | Test Fscore: {fscore*100:.2f}%"
    )

    return X_test, y_test, bow_nn.predict(test_iterator)


if __name__ == "__main__":
    main("../data/headline_data/headlines.conllu")
