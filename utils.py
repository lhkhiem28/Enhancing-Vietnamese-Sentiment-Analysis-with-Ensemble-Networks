import random
import numpy as np
import prettytable
from sklearn.model_selection import train_test_split

def train_eval_split(df, eval_size):
    train_df, eval_df = train_test_split(df, stratify=df["mapped_rating"], test_size=0.2, random_state=2020)

    train_counts, eval_counts = train_df["mapped_rating"].value_counts().sort_index().values, eval_df["mapped_rating"].value_counts().sort_index().values
    print("\nDataset Statistics:\n" + "===================")
    table = prettytable.PrettyTable(["Class", "Train", "Eval"])
    table.align["Class"], table.align["Train"], table.align["Eval"] = "l", "l", "l"
    class_names = ["neg", "pos"]
    for i in range(len(class_names)):
        table.add_row([class_names[i], train_counts[i], eval_counts[i]])
    print(table)

    return train_df, eval_df

def get_trainable_modules(model):
    trainable_modules = []
    for name, child in model.named_children():
        for p in child.parameters():
            if p.requires_grad:
                trainable_modules.append(name)
                break
    
    return trainable_modules