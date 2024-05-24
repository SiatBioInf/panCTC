
from torch import nn
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x)
        
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)
        
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)
      
        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)
        return x

if __name__ == '__main__':
    import torch
    device_ids = [Id for Id in range(torch.cuda.device_count())]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enconder=EncoderLayer(8,8000,4,0.5)
    enconder=enconder.to(device)
    data=torch.randn(1,1103,8)
    data=data.cuda()
    output=enconder(data)
    print(output.shape)

