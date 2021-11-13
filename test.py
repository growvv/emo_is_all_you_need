import torch
from transformers import BertTokenizer, BertModel
import ipdb
import config   


tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

x = model.embeddings.word_embeddings.weight[-1, :]

print(len(tokenizer))  # 28996
tokenizer.add_tokens(["NEW_TOKEN"])
print(len(tokenizer))  # 28997

model.resize_token_embeddings(len(tokenizer)) 
# The new vector is added at the end of the embedding matrix

print(model.embeddings.word_embeddings.weight[-1, :])
# Randomly generated matrix

with torch.no_grad():
    model.embeddings.word_embeddings.weight[-1, :] = torch.zeros([model.config.hidden_size])

print(model.embeddings.word_embeddings.weight[-1, :])
# outputs a vector of zeros of shape [768]

y = model.embeddings.word_embeddings.weight[-2, :]

print(x == y) # 会改变原来embedding weight 吗？ 不会
ipdb.set_trace()
