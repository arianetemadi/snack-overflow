# Simple BoW NN using the implementation from the lecture
# Source: https://github.com/tuw-nlp-ie/tuw-nlp-ie-2024WS/blob/main/lectures/05_Deep_learning_practical_lesson/deep_learning_practical_lesson_without_outputs.ipynb

import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import torch.optim as optim


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
        rounded_preds.cpu(), y.cpu()
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

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0
        self.model.eval()

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

    def training_loop(self, train_iterator, valid_iterator, epoch_number=15):
        train_losses = []
        valid_losses = []

        for epoch in range(epoch_number):
            train_loss, train_prec, train_rec, train_fscore = self.train(train_iterator)

            valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(
                valid_iterator
            )

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(
                f"\tTrain Loss: {train_loss:.3f} | Train Prec: {train_prec*100:.2f}% | Train Rec: {train_rec*100:.2f}% | Train Fscore: {train_fscore*100:.2f}%"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val Prec: {valid_prec*100:.2f}% | Val Rec: {valid_rec*100:.2f}% | Val Fscore: {valid_fscore*100:.2f}%"
            )
