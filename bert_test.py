from transformers import BertTokenizer, BertModel
import torch
import ipdb


name = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertModel.from_pretrained(name)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

# ipdb.set_trace()
print(outputs.last_hidden_state.shape)
print(outputs.pooler_output.shape)