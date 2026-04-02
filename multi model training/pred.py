"""
pred.py

Matthew Ma
Ethan Tang
Abbas Peermohammed
Cody Chen

=============================

Logistic regression predictor
"""

import numpy as np
import re
import csv

# Load model parameters 
params = np.load('lr_model_params.npz', allow_pickle=True)

# Unpack parameters
scaler_mean = params['scaler_mean'] 
scaler_scale = params['scaler_scale'] 
vocabulary = params['vocabulary'].item() 
target_classes = params['target_classes'] 
weights = params['weights'] 
intercept = params['intercept'] 

# Binary category lists 
room_categories = params['room_categories'].tolist()
who_categories = params['who_categories'].tolist()
season_categories = params['season_categories'].tolist()
numeric_cols = params['numeric_cols'].tolist()

# Helper functions 
def clean_emotion_intensity(x):
    if not x or x == '':
        return np.nan
    try:
        return float(x)
    except:
        return np.nan

def clean_count(x):
    if not x or x == '':
        return np.nan
    x = str(x).replace(',', '').strip()
    match = re.search(r'\d+', x)
    if match:
        return int(match.group())
    else:
        return np.nan

def clean_price(x):
    if not x or x == '':
        return np.nan
    x = str(x).lower()
    x = x.replace('$', '').replace(',', '').strip()
    multiplier = 1
    if 'million' in x:
        multiplier = 1_000_000
        x = x.replace('million', '')
    elif 'billion' in x:
        multiplier = 1_000_000_000
        x = x.replace('billion', '')
    elif 'k' in x or 'thousand' in x:
        multiplier = 1000
        x = x.replace('k', '').replace('thousand', '')
    numbers = re.findall(r'(\d+(?:\.\d+)?)', x)
    if numbers:
        return float(numbers[0]) * multiplier
    else:
        return np.nan

def split_multi(x):
    if not x or x == '':
        return []
    return [item.strip() for item in str(x).split(',')]

def likert_map(x):
    mapping = {
        "1 - Strongly disagree": 1,
        "2 - Disagree": 2,
        "3 - Neutral/Unsure": 3,
        "4 - Agree": 4,
        "5 - Strongly agree": 5
    }
    return mapping.get(x, 3) # Returns a default value of 3 if the value is missing

# Feature extraction for a single row 
def extract_features(row):
    # 1. Clean the numeric features following the same order in the training data
    num_vals = []
    for col in numeric_cols:
        if col == 'emotion_intensity':
            val = clean_emotion_intensity(row.get(col, ''))
        elif col == 'prominent_colours':
            val = clean_count(row.get(col, ''))
        elif col == 'objects_caught':
            val = clean_count(row.get(col, ''))
        elif col == 'price_willing':
            val = clean_price(row.get(col, ''))
        else:  # Likert columns
            val = likert_map(row.get(col, ''))
        num_vals.append(val)
    num_vals = np.array(num_vals, dtype=float)
    # Impute NaN with 0
    num_vals = np.nan_to_num(num_vals, nan=0.0)
    # Standardize
    num_scaled = (num_vals - scaler_mean) / scaler_scale

    # 2. Binary indicators for room, people, and season (Same order as training data)
    room_list = split_multi(row.get("If you could purchase this painting, which room would you put that painting in?", ""))
    room_bin = [1 if cat in room_list else 0 for cat in room_categories]
    who_list = split_multi(row.get("If you could view this art in person, who would you want to view it with?", ""))
    who_bin = [1 if cat in who_list else 0 for cat in who_categories]
    season_list = split_multi(row.get("What season does this art piece remind you of?", ""))
    season_bin = [1 if cat in season_list else 0 for cat in season_categories]
    binary_arr = np.array(room_bin + who_bin + season_bin, dtype=int)

    # 3. Create a bag of words using a saved vocabulary from training data for text features
    desc = row.get("Describe how this painting makes you feel.", "")
    food = row.get("If this painting was a food, what would be?", "")
    soundtrack = row.get("Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.", "")
    text = f"{desc} {food} {soundtrack}"
    tokens = text.lower().split()
    bow = np.zeros(len(vocabulary))
    for word in tokens:
        if word in vocabulary:
            bow[vocabulary[word]] += 1

    # Combine
    return np.concatenate([num_scaled, binary_arr, bow])

# Prediction for one row
def predict(row):
    x = extract_features(row)
    logits = x @ weights.T + intercept
    # Softmax (for numerical stability)
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    pred_idx = np.argmax(probs)
    return target_classes[pred_idx]

# Make a prediction for all rows
def predict_all(filename):
    """
    Read CSV file and return a list of predictions.
    """
    predictions = []
    with open(filename, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(predict(row))
    return predictions

