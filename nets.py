import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, embedding_matrix, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.2):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(1, n_filters, (K, embedding_matrix.shape[1])) for K in kernel_sizes])
        self.drop  = nn.Dropout(dropout_prob)
        self.extracter = nn.Linear(n_filters * len(kernel_sizes), 256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = embedding.unsqueeze(1)  

        convs = [conv(embedding) for conv in self.convs] 
        convs = [F.relu(conv).squeeze(3) for conv in convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  

        cat = torch.cat(pools, 1)
        fea = self.extracter(cat)
        out = self.drop(fea)  
        out = self.classifier(out)  

        return fea, out

class LSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.2):
        super(LSTM, self).__init__()        
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout_prob)
        self.extracter = nn.Linear(6*hidden_size, 256)
        self.classifier = nn.Linear(256, 1)
        
    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))
        lstm, _ = self.lstm(embedding)
        pooling = torch.cat((torch.mean(lstm, 1), torch.max(lstm, 1)[0]), 1)
        
        cat = torch.cat((pooling, lstm[:, -1, :]), 1)
        fea = self.extracter(cat)
        out = self.drop(fea)
        out = self.classifier(out)

        return fea, out

class GRU(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.2):
        super(GRU, self).__init__()        
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        
        self.gru  = nn.GRU(embedding_matrix.shape[1], hidden_size, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(dropout_prob)
        self.extracter = nn.Linear(6*hidden_size, 256)
        self.classifier = nn.Linear(256, 1)
        
    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(torch.unsqueeze(embedding, 0))
        gru, _  = self.gru(embedding)
        pooling = torch.cat((torch.mean(gru, 1), torch.max(gru, 1)[0]), 1)
        
        cat = torch.cat((pooling, gru[:, -1, :]), 1)
        fea = self.extracter(cat)
        out = self.drop(fea)
        out = self.classifier(out)

        return fea, out

class LSTMCNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.2):
        super(LSTMCNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_matrix.shape[1], self.hidden_size, bidirectional=True)
        self.drop = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(embedding_matrix.shape[1] + 2*self.hidden_size, self.hidden_size)
        self.extracter = nn.Linear(self.hidden_size, 256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        embedding = self.embedding(x).permute(1, 0, 2)

        lstm, _ = self.lstm(embedding)
        
        cat = torch.cat((lstm[:, :, :self.hidden_size], embedding, lstm[:, :, self.hidden_size:]), 2).permute(1, 0, 2)
        cat = torch.tanh(self.linear(cat)).permute(0, 2, 1)
        cat = F.max_pool1d(cat, cat.shape[2]).squeeze(2)
        fea = self.extracter(cat)
        out = self.drop(fea)
        out = self.classifier(out) 
        
        return fea, out

class GRUCNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.2):
        super(GRUCNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.gru  = nn.GRU(embedding_matrix.shape[1], self.hidden_size, bidirectional=True)
        self.drop = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(embedding_matrix.shape[1] + 2*self.hidden_size, self.hidden_size)
        self.extracter = nn.Linear(self.hidden_size, 256)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        embedding = self.embedding(x).permute(1, 0, 2)

        gru, _  = self.gru(embedding)
        
        cat = torch.cat((gru[:, :, :self.hidden_size], embedding, gru[:, :, self.hidden_size:]), 2).permute(1, 0, 2)
        cat = torch.tanh(self.linear(cat)).permute(0, 2, 1)
        cat = F.max_pool1d(cat, cat.shape[2]).squeeze(2)
        fea = self.extracter(cat)
        out = self.drop(fea)
        out = self.classifier(out) 
        
        return fea, out