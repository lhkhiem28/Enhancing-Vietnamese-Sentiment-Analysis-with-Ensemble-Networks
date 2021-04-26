import torch
import torch.nn as nn
import torch.nn.functional as F
import acts
from nets import *
from utils import *

class EnsembleLinear(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleLinear, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(num_models*featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        cat = torch.cat(
            (
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ), 1
        )
        out = self.linear(cat)
        out = self.drop(out)
        out = self.classifier(out)

        return out

class EnsembleAttention(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleAttention, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.se_module = nn.Sequential(
            nn.Linear(num_models*featrue_dim, (num_models*featrue_dim) // 8, bias=False),
            acts.Swish(),
            nn.Linear((num_models*featrue_dim) // 8, num_models*featrue_dim, bias=False),
            acts.Sigmoid(),
        )
        self.linear = nn.Linear(num_models*featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        cat = torch.cat(
            (
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ), 1
        )
        att = self.se_module(cat) * cat
        out = self.linear(att)
        out = self.drop(out)
        out = self.classifier(out)

        return out

class EnsembleSqueezeExcitation(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleSqueezeExcitation, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.se_module = nn.Sequential(
            nn.Linear(num_models, 3, bias=False),
            acts.Swish(),
            nn.Linear(3, num_models, bias=False),
            acts.Sigmoid(),
        )
        self.linear = nn.Linear(num_models*featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x): 
        stack = torch.stack(
            [
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ], -2
        )                                       
        pool = F.adaptive_avg_pool1d(stack, 1)  
        pool = pool.view(pool.size(0), -1)      
        
        se = self.se_module(pool)              
        se = se.unsqueeze(-1) * stack            
        se = se.view(se.size(0), -1)         
        out = self.linear(se)
        out = self.drop(out)
        out = self.classifier(out)

        return out

class EnsembleMoESigmoid(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleMoESigmoid, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.linear_gate = nn.Linear(num_models*featrue_dim, num_models)
        self.linear = nn.Linear(featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        cat = torch.cat(
            (
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ), 1
        )
        input_gate = cat.detach()
        output_gate = self.linear_gate(input_gate.float())
        output_gate = torch.sigmoid(output_gate)

        stack = torch.stack(
            [
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ], -2
        )
        out = torch.sum(output_gate.unsqueeze(-1) * stack, -2)
        out = self.linear(out)
        out = self.drop(out)
        out = self.classifier(out)

        return out

class EnsembleMoESoftmax(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleMoESoftmax, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.linear_gate = nn.Linear(num_models*featrue_dim, num_models)
        self.linear = nn.Linear(featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        cat = torch.cat(
            (
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ), 1
        )
        input_gate = cat.detach()
        output_gate = self.linear_gate(input_gate.float())
        output_gate = torch.softmax(output_gate, -1)

        stack = torch.stack(
            [
                self.textcnn(x)[0],
                self.lstm(x)[0],
                self.gru(x)[0],
                self.lstmcnn(x)[0],
                self.grucnn(x)[0],
            ], -2
        ) 
        out = torch.sum(output_gate.unsqueeze(-1) * stack, -2)
        out = self.linear(out)
        out = self.drop(out)
        out = self.classifier(out)

        return out

class EnsembleUniformWeight(nn.Module):
    def __init__(self, embedding_matrix, num_models, pretrained_weights, featrue_dim=256, dropout_prob=0.2):
        super(EnsembleUniformWeight, self).__init__()
        self.textcnn = TextCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.lstm = LSTM(embedding_matrix, featrue_dim=featrue_dim)
        self.gru = GRU(embedding_matrix, featrue_dim=featrue_dim)
        self.lstmcnn = LSTMCNN(embedding_matrix, featrue_dim=featrue_dim)
        self.grucnn = GRUCNN(embedding_matrix, featrue_dim=featrue_dim)

        self.textcnn.load_state_dict(torch.load(pretrained_weights["textcnn"]))
        self.lstm.load_state_dict(torch.load(pretrained_weights["lstm"]))
        self.gru.load_state_dict(torch.load(pretrained_weights["gru"]))
        self.lstmcnn.load_state_dict(torch.load(pretrained_weights["lstmcnn"]))
        self.grucnn.load_state_dict(torch.load(pretrained_weights["grucnn"]))

        for p in self.textcnn.parameters():
            p.requires_grad = False
        for p in self.lstm.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        for p in self.lstmcnn.parameters():
            p.requires_grad = False
        for p in self.grucnn.parameters():
            p.requires_grad = False

        self.linear = nn.Linear(featrue_dim, 1024)
        self.drop = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        textcnn = self.textcnn(x)[0]
        lstm = self.lstm(x)[0]
        gru = self.gru(x)[0]
        lstmcnn = self.lstmcnn(x)[0]
        grucnn = self.grucnn(x)[0]

        avg = (textcnn + lstm + gru + lstmcnn + grucnn) / 5
        out = self.linear(avg)
        out = self.drop(out)
        out = self.classifier(out)

        return out