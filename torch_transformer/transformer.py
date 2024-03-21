from . import DEVICE
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):

  def __init__(self, dim, dropout = 0.1, max_len = 5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1).to(DEVICE)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)).to(DEVICE)
    pe = torch.zeros(max_len, 1, dim).to(DEVICE)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)

class MultiHeadAttention(nn.Module):

  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__()
    self.dim = dim
    self.head_num = head_num
    self.linear_Q = nn.Linear(dim, dim, bias = False).to(DEVICE)
    self.linear_K = nn.Linear(dim, dim, bias = False).to(DEVICE)
    self.linear_V = nn.Linear(dim, dim, bias = False).to(DEVICE)
    self.linear = nn.Linear(dim, dim, bias = False).to(DEVICE)
    self.soft = nn.Softmax(dim = 3).to(DEVICE)
    self.dropout = nn.Dropout(dropout).to(DEVICE)

  def split_head(self, x):
    x = torch.tensor_split(x, self.head_num, dim = 2)
    x = torch.stack(x, dim = 1)
    return x

  def concat_head(self, x):
    x = torch.tensor_split(x, x.size()[1], dim = 1)
    x = torch.concat(x, dim = 3).squeeze(dim = 1)
    return x

  def forward(self, Q, K, V, mask = None):
    Q = self.linear_Q(Q)   #(BATCH_SIZE,word_count,dim)
    K = self.linear_K(K)
    V = self.linear_V(V)

    Q = self.split_head(Q)   #(BATCH_SIZE,head_num,word_count//head_num,dim)
    K = self.split_head(K)
    V = self.split_head(V)

    QK = torch.matmul(Q, torch.transpose(K, 3, 2))
    QK = QK/((self.dim//self.head_num)**0.5)

    if mask is not None:
      QK = QK + mask

    softmax_QK = self.soft(QK)
    softmax_QK = self.dropout(softmax_QK)

    QKV = torch.matmul(softmax_QK, V)
    QKV = self.concat_head(QKV)
    QKV = self.linear(QKV)
    return QKV

class FeedForward(nn.Module):

  def __init__(self, dim, hidden_dim = 512, dropout = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(dropout).to(DEVICE)
    self.linear_1 = nn.Linear(dim, hidden_dim).to(DEVICE)
    self.relu = nn.ReLU().to(DEVICE)
    self.linear_2 = nn.Linear(hidden_dim, dim).to(DEVICE)

  def forward(self, x):
    x = self.linear_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.linear_2(x)
    return x

################################################k
### エンコーダ・デコーダ（Transformer）
################################################

class EncoderBlock(nn.Module):

  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__()
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim]).to(DEVICE)
    self.layer_norm_2 = nn.LayerNorm([dim]).to(DEVICE)
    self.FF = FeedForward(dim).to(DEVICE)
    self.dropout_1 = nn.Dropout(dropout).to(DEVICE)
    self.dropout_2 = nn.Dropout(dropout).to(DEVICE)

  def forward(self, x):
    Q = K = V = x
    x = self.MHA(Q, K, V)
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_2(x)
    x = x + _x
    x = self.layer_norm_2(x)
    return x

class Encoder(nn.Module):

  def __init__(self, enc_vocab_size, dim, head_num, dropout = 0.1):
    super().__init__()
    self.dim = dim
    self.embed = nn.Embedding(enc_vocab_size, dim).to(DEVICE)
    self.PE = PositionalEncoding(dim)
    self.dropout = nn.Dropout(dropout).to(DEVICE)
    self.EncoderBlocks = nn.ModuleList([EncoderBlock(dim, head_num) for _ in range(6)])

  def forward(self, x):
    x = self.embed(x)
    x = x*(self.dim**0.5)
    x = self.PE(x)
    x = self.dropout(x)
    for i in range(6):
      x = self.EncoderBlocks[i](x)
    return x

class DecoderBlock(nn.Module):

  def __init__(self, dim, head_num, dropout = 0.1):
    super().__init__()
    self.MMHA = MultiHeadAttention(dim, head_num)
    self.MHA = MultiHeadAttention(dim, head_num)
    self.layer_norm_1 = nn.LayerNorm([dim]).to(DEVICE)
    self.layer_norm_2 = nn.LayerNorm([dim]).to(DEVICE)
    self.layer_norm_3 = nn.LayerNorm([dim]).to(DEVICE)
    self.FF = FeedForward(dim)
    self.dropout_1 = nn.Dropout(dropout).to(DEVICE)
    self.dropout_2 = nn.Dropout(dropout).to(DEVICE)
    self.dropout_3 = nn.Dropout(dropout).to(DEVICE)

  def forward(self, x, y, mask):
    Q = K = V = x
    x = self.MMHA(Q, K, V, mask)
    x = self.dropout_1(x)
    x = x + Q
    x = self.layer_norm_1(x)
    Q = x
    K = V = y
    x = self.MHA(Q, K, V)
    x = self.dropout_2(x)
    x = x + Q
    x = self.layer_norm_2(x)
    _x = x
    x = self.FF(x)
    x = self.dropout_3(x)
    x = x + _x
    x = self.layer_norm_3(x)
    return x

class Decoder(nn.Module):

  def __init__(self, dec_vocab_size, dim, head_num, dropout = 0.1):
    super().__init__()
    self.dim = dim
    self.embed = nn.Embedding(dec_vocab_size, dim).to(DEVICE)
    self.PE = PositionalEncoding(dim)
    self.DecoderBlocks = nn.ModuleList([DecoderBlock(dim, head_num) for _ in range(6)])
    self.dropout = nn.Dropout(dropout).to(DEVICE)
    self.linear = nn.Linear(dim, dec_vocab_size).to(DEVICE)

  def forward(self, x, y, mask):
    x = self.embed(x)
    x = x*(self.dim**0.5)
    x = self.PE(x)
    x = self.dropout(x)
    for i in range(6):
      x = self.DecoderBlocks[i](x, y, mask)
    x = self.linear(x)   #損失の計算にnn.CrossEntropyLoss()を使用する為、Softmax層を挿入しない
    return x

class Transformer(nn.Module):

  def __init__(self, enc_vocab_size, dec_vocab_size, dim, head_num):
    super().__init__()
    self.encoder = Encoder(enc_vocab_size, dim, head_num)
    self.decoder = Decoder(dec_vocab_size, dim, head_num)

  def forward(self, enc_input, dec_input, mask):
    enc_output = self.encoder(enc_input)

    output = self.decoder(dec_input, enc_output, mask)
    return output