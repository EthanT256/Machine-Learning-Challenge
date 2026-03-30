import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("ml_challenge_dataset.csv")

# split (90% train, 10% test)
train, test = train_test_split(data, test_size=0.1, random_state=42)

# save to CSV
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)