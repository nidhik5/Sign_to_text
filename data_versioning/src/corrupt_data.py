import pandas as pd
import numpy as np
import os

os.makedirs("data/corrupted", exist_ok=True)

train = pd.read_csv("data/processed/train_processed.csv")

labels = train.iloc[:, 0]
images = train.iloc[:, 1:].values

noise = np.random.normal(0, 0.2, images.shape)

corrupted = np.clip(images + noise, 0, 1)

corrupted_df = pd.DataFrame(corrupted)
corrupted_df.insert(0, "label", labels)

corrupted_df.to_csv(
    "data/corrupted/train_corrupted.csv",
    index=False
)

print("Corrupted dataset generated")