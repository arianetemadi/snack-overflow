import re
import numpy as np


def fit_patterns(headlines, patterns, label="1", verbose=True, num_examples=10):
    """
    Fit patterns to the input headlines for the given label.

    Args:
        headlines: List of headlines to consider.
        patterns (list of strings): regex patterns to try.
        label (str): "0" or "1", the class to use for the report.
        verbose (boolean): whether or not to print the report.
        num_examples (int): number of random sample headlines to show that conform to any pattern.

    Returns:
        tuple of precision, recall, and subset (list of all headlines that fit at least one pattern).
    """
    
    # filter headlines
    subset = []
    for headline in headlines:
        for pattern in patterns:
            text = headline[0].metadata["text"]
            match = re.match(pattern, text)
            if match:
                subset.append(headline)
                break
    
    # compute the precision and recall of this pattern
    tp = 0
    for headline in subset:
        if headline[0].metadata["class"] == label:
            tp += 1
    tp_fp = len(subset)
    tp_fn = 0
    for headline in headlines:
        if headline[0].metadata["class"] == label:
            tp_fn += 1

    if tp_fp > 0:
        precision = tp / tp_fp
    else:
        precision = 0.0
    recall = tp / tp_fn

    if verbose:
        print(f"-- class = {label}: precision={precision}, recall={recall}")
        print("-- examples that fit these patterns: ")
        if tp_fp > 0:
            for i in np.random.randint(0, len(subset), num_examples):
                print(subset[i][0].metadata["text"])
    
    return precision, recall, subset
