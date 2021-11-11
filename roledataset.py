from re import S
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pandas as pd
import config
from transformers import BertTokenizer
from utils import get_batch_sizes
import ipdb


class RoleDataset(Dataset):
    def __init__(self, tokenizer, max_len, mode='train'):
        super(RoleDataset, self).__init__()
        if mode == 'train':
            self.data = pd.read_csv('data/train.csv',sep='\t')
        else:
            self.data = pd.read_csv('data/test.csv',sep='\t')
        self.texts=self.data['text'].tolist()
        # ipdb.set_trace()
        self.labels=self.data[config.target_cols].to_dict('records')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.id = self.data['id'].tolist()

    def __getitem__(self, index):
        text = self.texts[index]  # 天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装，完全暴露在大雨之中。角色: o2'
        label = self.labels[index]  # {'love': 0, 'joy': 0, 'fright': 0, 'anger': 0, 'fear': 0, 'sorrow': 0}
        id = self.id[index]
        # ipdb.set_trace()

        encoding=self.tokenizer.encode_plus(text,
                                            add_special_tokens=True,
                                            truncation=True,
                                            padding = 'max_length',
                                            max_length=self.max_len,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',)

        sample = {
            'texts': text,
            'input_ids': encoding['input_ids'].flatten(), # [max_length]
            'attention_mask': encoding['attention_mask'].flatten(), # [max_length]
            'id': id,
        }

        for label_col in config.target_cols:
            sample[label_col] = torch.tensor(label[label_col], dtype=torch.float)
        return sample

    def __len__(self):
        return len(self.texts)


def create_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
    """
    Create dataloader for the given dataset accordcing to dynamic batch_size.
    """

    data_loader = DataLoader(dataset, batch_size=end-start, shuffle=False, num_workers=4)
    return data_loader


def all_data(dataset):
    """
    Return all data in the dataset.
    """
    data = []
    for index in range(len(dataset)):
        data.append(dataset.__getitem__(index))
    return data


if __name__ == "__main__":
    print("loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    print("loading finish!")

    trainset = RoleDataset(tokenizer, config.max_len, mode='train')
    data = all_data(trainset)

    batch_sizes = get_batch_sizes(file_path='data/train.csv')
    for start, end in batch_sizes:
        # print("len: ", end-start)
        batch_size = min(end-start, config.batch_size)
        train_loader = DataLoader(dataset=data[start:end], batch_size=batch_size)

        if len(train_loader) > 1:
            print(len(train_loader), start, end)
        # for step, sample in enumerate(train_loader):
        #     # ipdb.set_trace()
        #     print(step, sample['id'])
    
        # ipdb.set_trace()
        # print(trainset.__len__())  # 36612
        # print(trainset.__getitem__(0))


'''
sampel = {
        'texts': '天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装，完全暴露在大雨之中。角色: o2', 
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
