
data_split.py
Utility script to split image dataset into train / validation / test
---------------------------------------------------------------------------------

import os
import shutil
import random
import argparse

def split_data(src, dst, train_ratio, val_ratio, test_ratio):
    classes = ["healthy", "mci"]

    for cls in classes:
        images = os.listdir(os.path.join(src, cls))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            "train": images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, files in splits.items():
            split_dir = os.path.join(dst, split, cls)
            os.makedirs(split_dir, exist_ok=True)

            for f in files:
                shutil.copy(
                    os.path.join(src, cls, f),
                    os.path.join(split_dir, f)
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    split_data(args.src, args.dst, args.train, args.val, args.test)
    print("Data split completed successfully.")
