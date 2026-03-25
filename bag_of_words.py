import numpy as np
import pandas as pd
import re
from collections import Counter
from typing import Tuple

def create_bag_of_words_from_file(filename: str) -> Tuple[np.ndarray, list]:
    """
    Create binary bag of words from soundtrack and feeling description columns.
    
    Parameters:
        filename (str) : a string containing the path to a data set.

    Returns:
        X (np.ndarray) : Binary bag-of-words matrix with shape (n_responses, n_words)
        vocabulary (list) : List of all unique words found in the both columns
    """
    data = pd.read_csv(filename)
    
    soundtrack_col = 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'
    feeling_col = 'Describe how this painting makes you feel.'
    
    all_texts = []
    for i in range(len(data)):
        combined = str(data[soundtrack_col][i]) + " " + str(data[feeling_col][i])
        all_texts.append(combined)

    doc_words = []
    all_words = []
    
    for text in all_texts:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        
        doc_words.append(words)
        all_words.extend(words)
    
    vocabulary = sorted(set(all_words))
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    X = np.zeros((len(data), len(vocabulary)))
    
    for i, words in enumerate(doc_words):
        for word in set(words):  
            if word in word_to_idx:
                X[i, word_to_idx[word]] = 1
    
    return X, vocabulary
    
