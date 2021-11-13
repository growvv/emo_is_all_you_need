import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import config
from transformers import BertTokenizer
import ipdb

def create_dataloader(dataset, batch_size, shuffle=False):

    if shuffle:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, mode='train'):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv('data/train.csv',sep='\t')
        else:
            self.data = pd.read_csv('data/test.csv',sep='\t')
        self.text=self.data['text'].tolist()
        self.labels=self.data[config.target_cols].to_dict('records')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.id =self.data['id'].tolist()
        self.character = self.data['character'].tolist()

    def __getitem__(self, index):
        text=str(self.text[index])  # 天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装，完全暴露在大雨之中。角色: o2'
        label=self.labels[index]  # {'love': 0, 'joy': 0, 'fright': 0, 'anger': 0, 'fear': 0, 'sorrow': 0}
        id = self.id[index]
        character = self.character[index]

        encoding=self.tokenizer.encode_plus(text,
                                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                            truncation=True,
                                            padding = 'max_length',
                                            max_length=self.max_len,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',)


        # ipdb.set_trace()
        tokens = self.tokenizer.decode(encoding['input_ids'].flatten()).split(' ')
        pos = tokens.index(character)
        # print(tokens)
        assert tokens[pos] == character
        assert pos < self.max_len and pos >= 0
        sample = {
            'id': id,
            'text': text,
            'character': character,
            'pos': pos,
            'input_ids': encoding['input_ids'].flatten(), # [max_length]
            'attention_mask': encoding['attention_mask'].flatten(), # [max_length]
        }

        # ipdb.set_trace()
        labels = []
        for label_col in config.target_cols:
            labels.append(label[label_col])
        
        sample['labels'] = torch.tensor(labels, dtype=torch.float)
        # sample['labels'] = labels
        return sample

    def __len__(self):
        return len(self.text)


if __name__ == "__main__":
    print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    print("loading finish!")

    trainset = RoleDataset(tokenizer, config.max_len, mode='train')
    train_loader = create_dataloader(trainset, config.batch_size)
    
    # ipdb.set_trace()
    print(trainset.__len__())  # 36612
    print(len(train_loader))  # 36612/32=1250 ?
    print(trainset.__getitem__(0)) 


'''
sampel = {
        'text': '天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装，完全暴露在大雨之中。角色: o2', 
        'input_ids': torch.Tensor([  101,  1921,  4958,   678,  4708,  3274,  7433,  8024,   157,  8144,
         3633,  1762,  5314, 10905,  4959,  7433,  6132,  8024,   800,  5632,
         2346,  1316,  1372,  4959,  4708,  1296,  5946,  4638,  1092,  6163,
         8024,  2130,  1059,  3274,  7463,  1762,  1920,  7433,   722,   704,
          511,  6235,  5682,   131,   157,  8144,   102,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0]), 
        'attention_mask': torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0]), 
        'love': torch.Tensor(0), 
        'joy': torch.Tensor(0), 
        'fright': torch.Tensor(0), 
        'anger': torch.Tensor(0), 
        'fear': torch.Tensor(0), 
        'sorrow': torch.Tensor(0)
    }
'''