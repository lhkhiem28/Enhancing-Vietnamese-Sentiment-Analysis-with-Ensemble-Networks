import torch
from data import load_tokenizer, load_embedding
from train import train, train_ensemble
from test import test, test_ensemble
root = "F:/Sentiment-Classification-SOFSEM-2020/"
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
#         lr=1e-3,
#         weight_decay=1e-5,
#         epochs=20,
#         device=device
#     )
for model_type in ["textcnn", "lstm", "gru", "lstmcnn", "grucnn"]:
    test(
        root,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix,
        model_type=model_type,
        pretrained=root + "logs/" + model_type + ".pth",
        device=device
    )

"""Ensembles"""
num_models = 5
pretrained_weights = {
    "textcnn": root + "logs/textcnn.pth",
    "lstm": root + "logs/lstm.pth",
    "gru": root + "logs/gru.pth",
    "lstmcnn": root + "logs/lstmcnn.pth",
    "grucnn": root + "logs/grucnn.pth",
}
# for model_type in ["linear", "attention", "squeezeexcitation", "moesigmoid", "moesoftmax", "uniformweight"]:
#     train_ensemble(
#         root,
#         eval_size=0.2,
#         tokenizer=tokenizer,
#         embedding_matrix=embedding_matrix,
#         model_type=model_type,
#         num_models=num_models,
#         pretrained_weights=pretrained_weights,
#         lr=1e-5,
#         weight_decay=1e-5,
#         epochs=20,
#         device=device
#     )
for model_type in ["linear", "attention", "squeezeexcitation", "moesigmoid", "moesoftmax", "uniformweight"]:
    test_ensemble(
        root,
        tokenizer=tokenizer,
        embedding_matrix=embedding_matrix,
        model_type=model_type,
        num_models=num_models,
        pretrained_weights=pretrained_weights,
        pretrained=root + "logs/" + "ensemble_" + model_type + ".pth",
        device=device
    )