import time
import json
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
from data import load_tokenizer, load_embedding
from nets import TextCNN, LSTM, GRU, LSTMCNN, GRUCNN
from ensembles import EnsembleLinear, EnsembleSqueezeExcitation, EnsembleUniformWeight, EnsembleMoESigmoid, EnsembleMoESoftmax
from utils import *

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

def train(root_path, eval_size, tokenizer, embedding_matrix, model_type, lr, weight_decay, epochs, device):
    df = pd.read_csv(root_path + "dataset/aivivn/train.csv")
    train_df, eval_df = train_eval_split(df, eval_size=eval_size)

    train_tokenized = tokenizer.texts_to_sequences(train_df["discriptions"].astype(str))
    X_train, y_train = pad_sequences(train_tokenized, maxlen=100), train_df["mapped_rating"].values
    X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    eval_tokenized = tokenizer.texts_to_sequences(eval_df["discriptions"].astype(str))
    X_eval, y_eval = pad_sequences(eval_tokenized, maxlen=100), eval_df["mapped_rating"].values
    X_eval, y_eval = torch.tensor(X_eval, dtype=torch.long), torch.tensor(y_eval, dtype=torch.long)
    eval_dataset = TensorDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=256)

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

    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    print("\nStart training ...\n" + "==================")
    print("Epochs:", epochs)
    print()
    since = time.time()
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        head = "epoch {:2}/{:2}".format(epoch, epochs)
        print(head + "\n" + "-"*(len(head)))

        model.train()
        running_losses = 0.0
        running_labels = []
        running_scores = []
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)[1]
            scores = torch.sigmoid(outputs)
            losses = criterion(outputs, labels.unsqueeze(1))

            losses.backward()
            optimizer.step()

            running_losses += losses.item() * inputs.size(0)
            running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
            running_scores += list(scores.cpu().detach().numpy())
        
        epoch_loss = running_losses / len(train_dataset)
        epoch_acc = accuracy_score(running_labels, np.round(running_scores))
        epoch_f1 = f1_score(running_labels, np.round(running_scores))
        print("{} - loss: {:.4f} - acc: {:.4f} - f1: {:.4f}".format("train", epoch_loss, epoch_acc, epoch_f1))

        with torch.no_grad():
            model.eval()
            running_losses = 0.0
            running_labels = []
            running_scores = []
            for inputs, labels in tqdm.tqdm(eval_loader):
                inputs, labels = inputs.to(device), labels.float().to(device)

                outputs = model(inputs)[1]
                scores = torch.sigmoid(outputs)
                losses = criterion(outputs, labels.unsqueeze(1))

                running_losses += losses.item() * inputs.size(0)
                running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
                running_scores += list(scores.cpu().detach().numpy())
        
        epoch_loss = running_losses / len(eval_dataset)
        epoch_acc = accuracy_score(running_labels, np.round(running_scores))
        epoch_f1 = f1_score(running_labels, np.round(running_scores))
        print("{} - loss: {:.4f} - acc: {:.4f} - f1: {:.4f}".format("eval ", epoch_loss, epoch_acc, epoch_f1))

        scheduler.step(epoch_loss)

        if best_acc < epoch_acc:
            best_acc = epoch_acc
            best_f1 = epoch_f1
            best_epoch = epoch
            torch.save(model.state_dict(), root_path + "logs/{}.pth".format(model_type))

    time_elapsed = time.time() - since
    print("\nTraining time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Result: epoch: {:2} - acc: {:.4f} - f1: {:.4f}".format(best_epoch, best_acc, best_f1))

def train_ensemble(root_path, eval_size, tokenizer, embedding_matrix, model_type, num_models, pretrained_weights, lr, weight_decay, epochs, device):
    df = pd.read_csv(root_path + "dataset/aivivn/train.csv")
    train_df, eval_df = train_eval_split(df, eval_size=eval_size)

    train_tokenized = tokenizer.texts_to_sequences(train_df["discriptions"].astype(str))
    X_train, y_train = pad_sequences(train_tokenized, maxlen=100), train_df["mapped_rating"].values
    X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    eval_tokenized = tokenizer.texts_to_sequences(eval_df["discriptions"].astype(str))
    X_eval, y_eval = pad_sequences(eval_tokenized, maxlen=100), eval_df["mapped_rating"].values
    X_eval, y_eval = torch.tensor(X_eval, dtype=torch.long), torch.tensor(y_eval, dtype=torch.long)
    eval_dataset = TensorDataset(X_eval, y_eval)
    eval_loader = DataLoader(eval_dataset, batch_size=256)

    print("\nLoad model...\n" + "=============")
    print("Model: ENSEMBLE - {}".format(model_type.upper()))
    if model_type == "linear":
        model = EnsembleLinear(embedding_matrix, num_models, pretrained_weights)
    if model_type == "squeezeexcitation":
        model = EnsembleSqueezeExcitation(embedding_matrix, num_models, pretrained_weights)
    if model_type == "uniformweight":
        model = EnsembleUniformWeight(embedding_matrix, num_models, pretrained_weights)
    if model_type == "moesigmoid":
        model = EnsembleMoESigmoid(embedding_matrix, num_models, pretrained_weights)
    if model_type == "moesoftmax":
        model = EnsembleMoESoftmax(embedding_matrix, num_models, pretrained_weights)

    print("Trainable modules:", get_trainable_modules(model))

    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    print("\nStart training ...\n" + "==================")
    print("Epochs:", epochs)
    print()
    since = time.time()
    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = 1
    for epoch in range(1, epochs + 1):
        head = "Epoch    {:2}/{:2}".format(epoch, epochs)
        print(head + "\n" + "-"*(len(head)))

        model.train()
        running_losses = 0.0
        running_labels = []
        running_scores = []
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            scores = torch.sigmoid(outputs)
            losses = criterion(outputs, labels.unsqueeze(1))

            losses.backward()
            optimizer.step()

            running_losses += losses.item() * inputs.size(0)
            running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
            running_scores += list(scores.cpu().detach().numpy())
        
        epoch_loss = running_losses / len(train_dataset)
        epoch_acc = accuracy_score(running_labels, np.round(running_scores))
        epoch_f1 = f1_score(running_labels, np.round(running_scores))
        print("{} - loss: {:.4f} - acc: {:.4f} - f1: {:.4f}".format("Train", epoch_loss, epoch_acc, epoch_f1))

        with torch.no_grad():
            model.eval()
            running_losses = 0.0
            running_labels = []
            running_scores = []
            for inputs, labels in tqdm.tqdm(eval_loader):
                inputs, labels = inputs.to(device), labels.float().to(device)

                outputs = model(inputs)
                scores = torch.sigmoid(outputs)
                losses = criterion(outputs, labels.unsqueeze(1))

                running_losses += losses.item() * inputs.size(0)
                running_labels += list(labels.unsqueeze(1).data.cpu().numpy())
                running_scores += list(scores.cpu().detach().numpy())
        
        epoch_loss = running_losses / len(eval_dataset)
        epoch_acc = accuracy_score(running_labels, np.round(running_scores))
        epoch_f1 = f1_score(running_labels, np.round(running_scores))
        print("{} - loss: {:.4f} - acc: {:.4f} - f1: {:.4f}".format("Eval ", epoch_loss, epoch_acc, epoch_f1))
        print()
        scheduler.step(epoch_loss)

        if best_acc < epoch_acc:
            best_acc = epoch_acc
            best_f1 = epoch_f1
            best_epoch = epoch
            torch.save(model.state_dict(), root_path + "logs/ensemble_{}.pth".format(model_type))

    time_elapsed = time.time() - since
    print("\nTraining time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Result: epoch: {:2} - acc: {:.4f} - f1: {:.4f}".format(best_epoch, best_acc, best_f1))