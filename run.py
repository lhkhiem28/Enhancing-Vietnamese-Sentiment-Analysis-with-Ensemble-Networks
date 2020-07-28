import warnings
warnings.filterwarnings("ignore")

import torch
from data import load_tokenizer, load_embedding
from train import train, train_ensemble
from test import test, test_ensemble
root = "F:/Enhance-Vietnamese-Sentiment-Classification-with-Deep-Learning-and-Ensemble-Techniques/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nLoad Tokenizer...\n" + "=================")
tokenizer = load_tokenizer(root + "dataset/aivivn/")
print("Loaded Tokenizer")
print("\nLoad Embedding...\n" + "=================")
embedding_matrix = load_embedding(root + "embedding/cc.vi.300.vec", embedding_size=300, word_index=tokenizer.word_index)
print("Loaded Embedding")

"""Weak models"""
# for model_type in ["textcnn", "lstm", "gru", "lstmcnn", "grucnn"]:
#     train(
#         root,
#         eval_size=0.2,
#         tokenizer=tokenizer,
#         embedding_matrix=embedding_matrix,
#         model_type=model_type,
#         featrue_dim=512,
#         lr=1e-3,
#         weight_decay=1e-5,
#         epochs=20,
#         device=device,
#     )
for model_type in ["textcnn", "lstm", "gru", "lstmcnn", "grucnn"]:
    test(
        root,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix,
        model_type=model_type,
        featrue_dim=512,
        pretrained=root + "logs/singles/512/{}.pth".format(model_type),
        device=device,
    )

"""Ensembles"""
num_models = 5
pretrained_weights = {
    "textcnn": root + "logs/singles/512/textcnn.pth",
    "lstm": root + "logs/singles/512/lstm.pth",
    "gru": root + "logs/singles/512/gru.pth",
    "lstmcnn": root + "logs/singles/512/lstmcnn.pth",
    "grucnn": root + "logs/singles/512/grucnn.pth",
}
# for model_type in ["linear", "attention", "squeezeexcitation", "moesigmoid", "moesoftmax", "uniformweight"]:
#     train_ensemble(
#         root,
#         eval_size=0.2,
#         tokenizer=tokenizer,
#         embedding_matrix=embedding_matrix,
#         model_type=model_type,
#         featrue_dim=512,
#         num_models=num_models,
#         pretrained_weights=pretrained_weights,
#         lr=1e-5,
#         weight_decay=1e-5,
#         epochs=20,
#         device=device,
#     )
for model_type in ["linear", "attention", "squeezeexcitation", "moesigmoid", "moesoftmax", "uniformweight"]:
    test_ensemble(
        root,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix,
        model_type=model_type,
        featrue_dim=512,
        num_models=num_models,
        pretrained_weights=pretrained_weights,
        pretrained=root + "logs/ens_5/512/ensemble_{}.pth".format(model_type),
        device=device,
    )