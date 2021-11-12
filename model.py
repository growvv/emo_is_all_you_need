import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import pool
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
import config
import numpy as np
from gat import GAT, create_graph, draw_graph, draw_graph_2

class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, bert):
        super(EmotionClassifier, self).__init__()
        self.bert = bert
        self.gat = GAT(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sigmod = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, text, character):
        last_hidden_state, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict = False
        )
        # ipdb.set_trace()
        
        data = create_graph(text, character, pooled_output)
        print(data)
        # draw_graph(data.edge_index) # 好用一点
        # draw_graph_2(data) # 拉跨

        # ipdb.set_trace()
        pooled_output = self.gat(data.x, data.edge_index)  # pooled_output[31] 全0为啥

        out = self.fc(pooled_output)
        out = self.sigmod(out)
        
        
        return out*3

if __name__ == "__main__":
    base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
    model = EmotionClassifier(n_classes=6, bert=base_model)

    input_ids = torch.randint(1, 10000, (256,1)).squeeze(1).unsqueeze(0)
    attention_mask = torch.Tensor([1 for i in range(256)]).unsqueeze(0)
    text = "hello,world"
    # ipdb.set_trace()
    print(input_ids.shape)
    print(attention_mask.shape)
    output = model(input_ids, attention_mask)
    print(output.shape)
