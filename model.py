import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
import config
import numpy as np
from multi_head_attention import MultihHeadAttention

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, bert):
        super(EmotionClassifier, self).__init__()
        self.bert = bert
        self.out_love = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.self_attention = MultihHeadAttention(4, 1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        love = self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)
        
        # ipdb.set_trace()
        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }
        # x = torch.cat((love, joy), 1)
        # x = torch.cat((x, fright), 1)
        # x = torch.cat((x, anger), 1)
        # x = torch.cat((x, fear), 1)
        # x = torch.cat((x, sorrow), 1)

        # x = self.self_attention(x, x, x)

        # return x

if __name__ == "__main__":
    base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
    model = EmotionClassifier(n_classes=4, bert=base_model)

    input_ids = torch.randint(1, 10000, (128,1)).squeeze(1).unsqueeze(0)
    attention_mask = torch.Tensor([1 for i in range(128)]).unsqueeze(0)
    # ipdb.set_trace()
    print(input_ids.shape)
    print(attention_mask.shape)
    output = model(input_ids, attention_mask)
    print(output)