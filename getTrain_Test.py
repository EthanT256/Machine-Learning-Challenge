import pandas as pd

# Load your CSV
df = pd.read_csv("ml_challenge_dataset.csv")  # replace with your file name

# Column containing unique IDs
id_col = "id"  # replace with your actual ID column name

# Get unique IDs and sort if you want "first 90%" in order
unique_ids = df["unique_id"].unique()

# Split point
split_idx = int(len(unique_ids) * 0.9)
train_ids = unique_ids[:split_idx]
test_ids = unique_ids[split_idx:]

# Create train and test sets
train_df = df[df["unique_id"].isin(train_ids)]
test_df  = df[df["unique_id"].isin(test_ids)]

# Save to CSV if needed
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Training set: {len(train_df)} rows")
print(f"Test set: {len(test_df)} rows")