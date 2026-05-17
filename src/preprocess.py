import os
import pandas as pd

os.makedirs("data/processed", exist_ok=True)

train = pd.read_csv("data/raw/sign_mnist_train.csv")
test = pd.read_csv("data/raw/sign_mnist_test.csv")

# Convert all columns to float32 first
train = train.astype("float32")
test = test.astype("float32")

# Normalize pixel values
train.iloc[:, 1:] /= 255.0
test.iloc[:, 1:] /= 255.0

# Save outputs
train.to_csv("data/processed/train_processed.csv", index=False)
test.to_csv("data/processed/test_processed.csv", index=False)

print("Preprocessing complete")