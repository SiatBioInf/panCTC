import torch
from torch import nn
import torch.nn.functional as F

from blocks.encoder_layer import EncoderLayer


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_prob):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob) for _ in range(n_layers)])
    def forward(self, x):
        for layer_module in self.layers:
            x = layer_module(x)
               
        return x


# Model2 to predict the cancer type
class Net_multi(nn.Module):
    def __init__(self, input_size, hidden_size, n_input, d_model, ffn_hidden, 
                 n_head, n_layers, drop_prob, n_hidden, n_out=12):
        super(Net_multi, self).__init__()
        
        self.mlp = MLP(input_size, hidden_size, n_input, drop_prob=drop_prob)
        self.linear0 = nn.Linear(1, d_model)
        self.encoder = Encoder(d_model=d_model, ffn_hidden=ffn_hidden,
                           n_head=n_head, n_layers=n_layers, drop_prob=drop_prob)
        self.dropout = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(d_model, 1)
        self.linear2 = nn.Linear(n_input, n_hidden)
        self.linear3 = nn.Linear(n_hidden, n_out)
        self.batch_norm  = nn.BatchNorm1d(n_hidden, eps=0.001, momentum=0.01)
        self.act = nn.LeakyReLU(0.2)

        
    def forward(self, x):
        x = self.mlp(x)
        x_ = x.unsqueeze(-1)
        x_ = self.linear0(x_)
        x = self.encoder(x_)
        x = self.linear1(x)
        x = x.squeeze(dim=-1) # [batch_size, n_input]
        x = self.dropout(x)
        x = self.act(self.batch_norm(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
    def predict(self, x):
        out = self.forward(x)
        ps = F.softmax(out, dim=1)
        top_p, top_class = ps.topk(3, dim=1)
        return top_class, top_p


if __name__ == '__main__':
    
    input_size = 200
    hidden_size = 64
    n_input = 16
    d_model = 4 
    ffn_hidden = 4
    n_head = 2 
    n_layers = 1
    drop_prob = 0.1
    n_hidden = 8

    x = torch.randn(20, input_size)
    net = Net_multi(input_size, hidden_size, n_input, d_model, ffn_hidden, 
                 n_head, n_layers, drop_prob, n_hidden)
    print(net(x).shape)
