import time
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from data import load_tokenizer, load_embedding
from nets import TextCNN, LSTM, GRU, LSTMCNN, GRUCNN
from ensembles import EnsembleLinear, EnsembleAttention, EnsembleSqueezeExcitation, EnsembleMoESigmoid, EnsembleMoESoftmax, EnsembleUniformWeight
from utils import *

def test(root_path, tokenizer, embedding_matrix, model_type, pretrained, device):
    test_df = pd.read_csv(root_path + "dataset/aivivn/test.csv")

    test_tokenized = tokenizer.texts_to_sequences(test_df["discriptions"].astype(str))
    X_test, y_test = pad_sequences(test_tokenized, maxlen=100), test_df["mapped_rating"].values
    X_test, y_test = torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256)

    print("\nLoad model...\n" + "=============")
    print("Model: {}".format(model_type.upper()))
    if model_type == "textcnn":
        model = TextCNN(embedding_matrix)
    if model_type == "lstm":
        model = LSTM(embedding_matrix)
    if model_type == "gru":
        model = GRU(embedding_matrix)
    if model_type == "lstmcnn":
        model = LSTMCNN(embedding_matrix)
    if model_type == "grucnn":
        model = GRUCNN(embedding_matrix)

    model.load_state_dict(torch.load(pretrained))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        running_labels = []
        running_scores = []
        for inputs, labels in tqdm.tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs)[1]
            scores = torch.sigmoid(outputs)

            running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
            running_scores += list(scores.cpu().detach().numpy())

    acc = accuracy_score(running_labels, np.round(running_scores))
    f1  = f1_score(running_labels, np.round(running_scores))
    auc = roc_auc_score(running_labels, running_scores)
    print("{} - acc: {:.4f} - f1: {:.4f} - auc: {:.4f}".format("test ", acc, f1, auc))

def test_ensemble(root_path, tokenizer, embedding_matrix, model_type, num_models, pretrained_weights, pretrained, device):
    test_df = pd.read_csv(root_path + "dataset/aivivn/test.csv")

    test_tokenized = tokenizer.texts_to_sequences(test_df["discriptions"].astype(str))
    X_test, y_test = pad_sequences(test_tokenized, maxlen=100), test_df["mapped_rating"].values
    X_test, y_test = torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256)

    print("\nLoad model...\n" + "=============")
    print("Model: ENSEMBLE - {}".format(model_type.upper()))
    if model_type == "linear":
        model = EnsembleLinear(embedding_matrix, num_models, pretrained_weights)
    if model_type == "attention":
        model = EnsembleAttention(embedding_matrix, num_models, pretrained_weights)
    if model_type == "squeezeexcitation":
        model = EnsembleSqueezeExcitation(embedding_matrix, num_models, pretrained_weights)
    if model_type == "moesigmoid":
        model = EnsembleMoESigmoid(embedding_matrix, num_models, pretrained_weights)
    if model_type == "moesoftmax":
        model = EnsembleMoESoftmax(embedding_matrix, num_models, pretrained_weights)
    if model_type == "uniformweight":
        model = EnsembleUniformWeight(embedding_matrix, num_models, pretrained_weights)

    print("Trainable modules:", get_trainable_modules(model))
    model.load_state_dict(torch.load(pretrained))
    model = model.to(device)

    with torch.no_grad():
        model.eval()
        running_labels = []
        running_scores = []
        for inputs, labels in tqdm.tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs)
            scores = torch.sigmoid(outputs)

            running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
            running_scores += list(scores.cpu().detach().numpy())

    acc = accuracy_score(running_labels, np.round(running_scores))
    f1  = f1_score(running_labels, np.round(running_scores))
    auc = roc_auc_score(running_labels, running_scores)
    print("{} - acc: {:.4f} - f1: {:.4f} - auc: {:.4f}".format("test ", acc, f1, auc))