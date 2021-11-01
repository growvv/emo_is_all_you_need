import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
import ipdb
from transformers.file_utils import is_pandas_available

def attention(query, key, value, mask=None, dropout=None):
    # 取query的最后一维，即embedding的维数
    d_k = query.size(-1)  
    #按照注意力公式，将query与key的转置相乘，这里面key是将最后两个维度进行转置，再除以缩放系数得到注意力得分张量scores
    # 如果query是[len, embed], 那么socres是[len, len]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # mask(也是[len, len]) 与 score 每个位置一一比较，如果mask[i][j]为0，则将scores[i][j]改为-1e9
        # 负很大的数，在softmax的相当于没有
        scores = scores.masked_fill(mask==0, -1e9)

    # 对最后一维进行softmax
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    # 最后，根据公式将p_attn与value张量相乘获得最终的query注意力表示，同时返回权重
    return torch.matmul(scores, value), scores


class MultihHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0):
        super(MultihHeadAttention, self).__init__()
        # 判断h是否能被d_model整除，这是因为我们之后要给每个头分配等量的词特征
        assert d_model % h == 0
        #得到每个头获得的分割词向量维度d_k
        self.d_k = d_model // h
        self.h = h

        self.w_key = nn.Linear(d_model, d_model)
        self.w_query = nn.Linear(d_model, d_model)
        self.w_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.atten = None  # 返回的attention张量，现在还没有，保存给可视化使用

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) # head导致query等多了一维
        
        # ipdb.set_trace()
        batch_size = query.size(0)
        query = self.w_query(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.w_key(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_value(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        x, self.atten = attention(query, key, value, mask, self.dropout)
        

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.fc_out(x)


if __name__ == "__main__":
    batch_size = 2
    len = 100
    d_model = 256
    head = 8
    query = torch.randn(batch_size, len, d_model)  # (batch_size, len, d_model)
    key = torch.randn(batch_size, len, d_model)
    value = torch.randn(batch_size, len, d_model)
    
    # mask = torch.randn(2, 100, 100)
    mask = torch.tril(torch.ones((2, 100, 100)))

    multi_attn = MultihHeadAttention(d_model, head)
    output = multi_attn(query, key, value, mask)

    assert output.shape == torch.Size([2, 100, 256])
    print(output.shape)


