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
- [`src/preprocessing.py`](src/preprocessing.py) - applies the pretrained `en_core_web_sm` model to tokenize, and lemmatize each of the headlines and saves the preprocessed sentences in the CoNLL format as the [preprocessed dataset](data/headline_data/headlines.conllu)
- [`notebooks/M1-data_overview.ipynb`](notebooks/M1-data_overview.ipynb) - analysis of the dataset
- [`data/headline_data`](data/headline_data) - contains the raw as well as preprocessed newspaper headlines dataset with the sarcasm class labeling

# Milestone 2
**Objective**: Implement multiple baseline solutions (DL & on-DL) to solve text classification task and provide quantitative and qualitative results discussion.

## Files
- [`notebooks/M2-NaiveBayes.ipynb`](notebooks/M2-NaiveBayes.ipynb) - applies the Naive Bayes model (with Bag of Words) to solve text classification task.
- [`notebooks/M2-dBERT.ipynb`](notebooks/M2-dBERT.ipynb) - applies 'distilbert-base-uncased' (lightweight alternative to BERT) model to solve text classification task. 
- [`notebooks/M2-LR.ipynb`](notebooks/M2-LR.ipynb) - applies Logistic Regression to the headline data.
- [`notebooks/M2-NN.ipynb`](notebooks/M2-NN.ipynb) - applies a simple one layer Neural Network (with Bag of words) to solve text classification task.

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


### Neural Network with Bag of Words

1. Setup:

Simple one linear layer NN with the log softmax activation function.
* Epochs = 18 (with early stopping criteria)
* Learning Rate = 1e-3
* Optimizer: Adam
* Loss function: negative log likelihood loss

2. Quantitative results on the test dataset:

- Precision: 0.8444
- Recall: 0.8288
- F1-Score: 0.8343

3. Qualitative analysis:

First, looking at the false negatives, so the headlines that contain sarcasm but are not classified as containing sarcasm, we note that these headlines don't particularly contain any words that would be indicative of sarcasm but in the context of the sentence as well as the social context they can be understood as sarcastic, however using a simple bag of words representation does not suffice to capture such phenomenons. 

On the other hand, the headlines marked as sarcastic but containing no sarcasm often are relatively hard to distinguish, because since we are using only a bag of words representation, the words that occur here might be more characteristic of sarcastic headlines like swear words or superlatives.  

### Naive Bayes with Bag of Words
1. Setup:

The Naive Bayes model with Bag of Words.
Supports any range of ngrams as the features to count.

2. Quantitative results on the test dataset (all numbers are macro averages of the two classes):

- Unigrams only:
    - Precision: 0.84
    - Recall: 0.84
    - F1-score: 0.84

- Bigrams only
    - Precision: 0.80
    - Recall: 0.79
    - F1-score: 0.79

- Unigrams, bigrams, and trigrams together:
    - Precision: 0.85
    - Recall: 0.85
    - F1-score: 0.85

3. Qualitative analysis:

- There are certain artifacts in the dataset. Some words appear much more frequently in one class without any relation to the notion of sarcasm. For instance, the word `trump` appears way more in non-sarcastic headlines. As a result, the model has a hard time detecting sarcastic headlines that include this word.
- There are certain patterns of joke that the sarcastic news source uses a lot. They are like templates. Therefore, certain words in these patterns are constant among many sarcastic headlines. This is not an artifact and helps our models to detect sarcasm, though only as long as we focus on this news source. It probably will not generalize as well to other sarcastic datasets. Example words include `nation`, `dad`, `study`, and `local`.
- Then there are swear words like `shit`. Since these will almost never appear in a non-sarcastic headline, they are useful for the model.
- There are smaller effects too. For instance, the word `three` appears noticeably more often in sarcastic headlines. We believe this is because sarcastic headlines are often fake, and we prefer to use the number three a lot when writing fake sentences. Number two would be too small, and numbers four and above might be too large. `three` is round and it is just enough. Saying `three months`, `three colors`, `three objects`, etc. sounds better compared to other numbers.

For a more detailed analysis with example sentences, refer to the notebook ([`notebooks/M2-NaiveBayes.ipynb`](notebooks/M2-NaiveBayes.ipynb)).

# Milestone 3 

**Objective**: Apply more advanced approaches to improve the performance over the baselines. The approaches included adding more data and extarcting feastures from syntactic analysis to build an ensamble model.

## Files
- [`notebooks/M3-analysis_tweets.ipynb`](notebooks/M3-analysis_tweets.ipynb) - analyzes the performance of the Naive Bayes baseline on new unseen data - tweets.
- [`notebooks/M3-chatgpt_headlines.ipynb`](notebooks/M3-chatgpt_headlines.ipynb) - analyzes the performance of the Naive Bayes baseline on new (unseen) headline data generated by ChatGPT. 
- [`notebooks/M3-patterns.ipynb`](notebooks/M3-patterns.ipynb) - applies regex pattern matching to uncover patterns in the headline data.
- [`notebooks/M3-syntactic_feature_sel.ipynb`](notebooks/M3-syntactic_feature_sel.ipynb) - peforms feature extraction from the syntactic analysis of the headlines data.
- [`notebooks/M3-voting_nb.ipynb`](notebooks/M3-voting_nb.ipynb) - applies the custom voting classifier to the various datasets.

### Milestone 3 - discussion

With the approaches that we have tried, the only improvement in the performance we saw applying the Naive Bayes baseline to the new data generated by ChatGPT. On the other hand, we were able to uncover some patterns in the data and get a better understanding of the impact of the writing style in the headlines on the classification. Although the voting classifier did not yield better performance than the baseline, we explored the various features that can be extracted from the syntactic analysis of the data and how these differ based on the model choice. Ultimately, we conclude that for a significant improvement of the baseline, we would have to incorporate external real-world knowledge into the classification process. 