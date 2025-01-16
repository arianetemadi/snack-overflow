# About the Datasets

## 1. Main dataset
**Name**: News-Headlines-Dataset-For-Sarcasm-Detection

**Authors**: R. Misra, A. Prahal, J. Grover

**Source**: [GitHub](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection)

**File**: [Sarcasm_Headlines_Dataset.json](Sarcasm_Headlines_Dataset.json)

## Description
The dataset contains around 28 thousand entries collected from newspaper article headlines of two websites [*TheOnion*](https://www.theonion.com/) and [*HuffPost*](https://www.huffingtonpost.com/). Each headline is further annotated with a binary class label describing whether it is sarcastic or not. The labelling corresponds to the source of the article headline, as *TheOnion* produces sarcastic headlines and *HuffPost* is a serious news publishing website.

## Preprocessing
Furthermore, this folder contains the output of the preprocessing applied to the raw data stored in the [dataset.conllu](dataset.conllu) file. Which was produced by the following script
```
python preprocessing.py Sarcasm_Headlines_Dataset.json dataset.conllu
```

## 1. Test dataset
**Name**: SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic

**Authors**:  I. Abu Farha, S. V. Oprea, S. Wilson, and W. Magdy

**Source**: [GitHub](https://github.com/iabufarha/iSarcasmEval/blob/main/train/train.En.csv)

**Files**: 
- [tweets.csv](tweets.csv) - original raw data
- [tweets.json](tweets.json) - raw data transformed into `.json` and sanitized using the `preprocess_tweets()` function from [src/preprocessing.py](preprocessing.py) 

## Description
The dataset contains tweets anottated by the authors to convey whether these are intentioanlly sarcastic or not. The dataset is imbalanced with a 1:4 ratio between sarcastic and non-sarcastic instances.

## Preprocessing
The data was preprocessed using the script [src/preprocessing.py](preprocessing.py) with the follwing command
```
python preprocessing.py tweets.json tweets.conllu
```