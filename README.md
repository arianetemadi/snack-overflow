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
    - ??

### DistilBERT results


