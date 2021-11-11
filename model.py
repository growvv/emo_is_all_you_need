import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
import config
import numpy as np
from multi_head_attention import MultihHeadAttention
from torchsummary import summary


class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, bert):
        super(EmotionClassifier, self).__init__()
        self.bert = bert
        self.out_love = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_joy = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes) #768
        # self.self_attention = MultihHeadAttention(n_classes, 1)
        # self.self_attention_2 = MultihHeadAttention(self.bert.config.hidden_size, 8)

    def forward(self, input_ids, attention_mask):
        # ipdb.set_trace()
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False,
        )
        ipdb.set_trace()
        # pooled_output = self.self_attention_2(pooled_output, pooled_output, pooled_output)
        love = self.out_love(pooled_output).unsqueeze(1)
        joy = self.out_joy(pooled_output).unsqueeze(1)
        fright = self.out_fright(pooled_output).unsqueeze(1)
        anger = self.out_anger(pooled_output).unsqueeze(1)
        fear = self.out_fear(pooled_output).unsqueeze(1)
        sorrow = self.out_sorrow(pooled_output).unsqueeze(1)

        # ipdb.set_trace()
        x = torch.cat((love, joy), 1)
        x = torch.cat((x, fright), 1)
        x = torch.cat((x, anger), 1)
        x = torch.cat((x, fear), 1)
        x = torch.cat((x, sorrow), 1)

        # ipdb.set_trace()
        # x = self.self_attention(x, x, x)
        # ipdb.set_trace()

        return x
        
        # ipdb.set_trace()
        # return {
        #     'love': love, 'joy': joy, 'fright': fright,
        #     'anger': anger, 'fear': fear, 'sorrow': sorrow,
        # }

if __name__ == "__main__":
    base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
    model = EmotionClassifier(n_classes=4, bert=base_model)
    # print(model)

    # input_ids = torch.randint(1, 10000, (8, 128))
    # attention_mask = torch.ones((8, 128))
    input_ids = torch.tensor([[3, 4, 5, 3]])
    attention_mask = torch.tensor([[1, 1, 1, 1]])
    # ipdb.set_trace()
    print(input_ids.shape)
    print(attention_mask.shape)
    output = model(input_ids, attention_mask)
    # print(output)
    # print(output.shape)
    # summary(model, input_ids, attention_mask)