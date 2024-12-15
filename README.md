# snack-overflow
Sarcasm Detection: Group project of the NLP course

Team members: Terezia Olsiakova, Maximilian Scheiblauer, Viktoriia Ovsianik, Arian Etemadi

# Enviroment Setup
The required dependencies can be installed and setup using conda as described in this section, taken from [spaCy](https://spacy.io/usage).

```
conda create -n venv
conda activate venv
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
```

To install, run the following in the root directory:

```
pip install -e .
```

# Milestone 1
**Objective**: Apply standard text preprocessing methods to the newspaper headlines and perform analysis to gain insights about the data.

## Files
- [`src/preprocessing.py`](src/preprocessing.py) - applies the pretrained `en_core_web_sm` model to tokenize, and lemmatize each of the headlines and saves the preprocessed sentences in the CoNLL format as the [preprocessed dataset](Data/dataset.conllu)
- [`notebooks/milestone1.ipynb`](notebooks/milestone1.ipynb) - analysis of the dataset
- [`data/`](data/) - contains the raw as well as preprocessed newspaper headlines dataset with the sarcasm class labeling

# Milestone 2
**Objective**: Implement multiple baseline solutions (DL & on-DL) to solve text classification task and provide quantitative and qualitative results discussion.

## Files
- [`notebooks/M2-NaiveBayes.ipynb`](notebooks/M2-NaiveBayes.ipynb) - applies BoW model to solve text classification task.
- [`notebooks/M2-dBERT.ipynb`](notebooks/M2-dBERT.ipynb) - aapplies 'distilbert-base-uncased' (lightweight alternative to BERT) model to solve text classification task. 

## Milestone 2 - discussion

For the second milestone we decided to implement 4 different baselines:
- Non-DL baselines:
    - Multinomial NB (BoW model)
    - Logistic Regression
-   DL baselines:
    - DistilBERT
    - Neural Network with Bag of Words



### DistilBERT

1. Setup:

**DistilBERT** is a smaller, faster, and more efficient version of BERT that retains most of its language understanding capabilities.
* Epochs = 10 (with early stopping criteria)
* Learning Rate = 5e-5
* Optimizer: AdamW (variant of Adam optimizer that helps to prevent overfitting)
* Loss function: BCEWithLogitsLoss (suitable for binary classification tasks)
* Dropout Rate = 0.25 (helps to prevent overfitting)

2. Quantitative results on the test dataset:

- Accuracy: 0.8377
- Precision: 0.8300
- Recall: 0.8316
- F1-Score: 0.8308
- Confusion Matrix: 

True Negatives (TN): 1883
False Positives (FP): 350
False Negatives (FN): 346
True Positives (TP): 1709

* Overall model *Accuracy* on the test set is 0.84, meaning the model performs quite well. However, since the task of binary classification is relatively simple, it is not surprising that the results of a deep learning approach are comparable with simpler, non-deep learning approaches. This suggests that for this particular task, it might make sense to stick to less computationally intensive and easier-to-interpret models.

* Model predicts "Sarcastic" class correctly 84.51% of the time (*precision*), and out of all the actual "Sarcastic" instances, it correctly identifies 83.16% (*recall*).

* The model tends to make more false positives (350) than false negatives (346), indicating that it is slightly better at detecting non-sarcastic instances. However, the difference is not large, meaning the model's errors are fairly balanced. When analyzing the probability distribution for misclassified cases, we observed that the model tends to make most mistakes near the threshold boundary. Therefore, we can try to improve performance by experimenting with different thresholds.

3. Qualitative analysis:

As a part of qualitative analysis we printed out all cases of misclassification, additionally, using 'transformers-interpret' library for incorrectly classified sarcastic sentences we pribnted out scores for each tokens to get more insight into how model makes predictions. We observed that:

* Tokens like new, man, woman, people, local, and social show up in the negative list. These words may often appear in factual or neutral contexts, causing the model to predict "non-sarcastic."

* Words such as trump, you, your, how, country, and campaign are more frequent in the positive list. These tokens often occur in sarcastic contexts due to their association with emotionally or politically charged topics.

* After analysis of incorrectly classified sentences it became clear that sometimes sarcastic sentences might rely on tone, humor, or implicit contradictions, which are hard for the model to detect.

4. Results discussion: 

* Initially, we aimed to use BERT as part of our deep learning solution. While the test set accuracy was high (0.93), the validation loss consistently increased with each epoch, indicating overfitting. Despite experimenting with different parameter combinations, we were unable to mitigate this issue. As a result, we decided to transition to a simpler BERT-based model, DistilBERT, which has 40% fewer parameters than BERT. This reduced complexity makes it easier to fine-tune and less prone to overfitting.

* For the next milestone, we plan to focus on either improving the performance of DistilBERT or revisiting the BERT model with optimized parameter configurations to achieve high quality without overfitting.


