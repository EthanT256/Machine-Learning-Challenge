import numpy as np
import pandas as pd
import re
from collections import Counter
from typing import Tuple

def create_bag_of_words_from_file(filename: str) -> Tuple[np.ndarray, list]:
    """
    Create binary bag of words.
    
    Parameters:
        filename (str) : a string containing the path to a data set.

    Returns:
        X (np.ndarray) : Binary bag-of-words matrix with shape (n_responses, n_words)
                         Entry = 1 if word appears in response, 0 otherwise
        vocabulary (list) : List of all unique words found in the soundtrack column
    """
    data = pd.read_csv(filename)
    
    soundtrack_col = 'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.'
    texts = data[soundtrack_col].fillna('').values # fill empty cells with the  empty string 
    
    doc_words = []
    all_words = []
    
    for text in texts:
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text) # remove special characters
        words = text.split()
        
        doc_words.append(words)
        all_words.extend(words)
    
    vocabulary = sorted(set(all_words))
    
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    X = np.zeros((len(texts), len(vocabulary)))
    
    for i, words in enumerate(doc_words):
        for word in set(words):  
            if word in word_to_idx:
                X[i, word_to_idx[word]] = 1
    
    return X, vocabulary

