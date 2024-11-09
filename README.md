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

To install the other required packages run:

```
pip install -r requirements_small.txt
```

# Milestone 1
**Objective**: Apply standard text preprocessing methods to the newspaper headlines and perform analysis to gain insights about the data.

## Files
- [`src/conll_converter.py`](src/conll_converter.py) - applies the pretrained `en_core_web_sm` model to tokenize, and lemmatize each of the headlines and saves the preprocessed sentences in the CoNLL format as the [preprocessed dataset](Data/dataset.conllu)
- [`Notebooks/data_checkout.ipynb`](Notebooks/data_checkout.ipynb) - analysis of the dataset
- [`Data/`](Data/) - contains the raw as well as preprocessed newspaper headlines dataset with the sarcasm class labeling