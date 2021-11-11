from tqdm import tqdm 
import pandas as pd
import ipdb
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt 


with open('huigui_ernie.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    data = list()
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        data.append(sp)

train = pd.DataFrame(data)
train.columns = ['id', 'emotions']

# print(train.head())
# print(len(train))
# print(train.shape)


for i in range(len(train)):
    emos = train.emotions[i].split(',')
    # ipdb.set_trace()
    tmp_emo = []
    for emo in emos:
        emo = float(emo)
        if emo < 0.2:
            emo = 0
        if emo > 3.0:
            emo = 3
        if emo > 0.9 and emo < 1.2:
            emo = 1
        if emo > 1.9 and emo < 2.2:
            emo = 2
        tmp_emo.append(str(emo))
    emos = ','.join(tmp_emo)
    train.emotions[i] = emos

# print(train.head(10))
# train['emotions'] = train['emotions'].apply(lambda x: [int(float(_i)+0.5) for _i in x.split(',')])
print(train.head())

# train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
# print(train.head(100))

# print(train['love'].value_counts())
# print(train['joy'].value_counts())
# print(train['fright'].value_counts())
# print(train['anger'].value_counts())
# print(train['fear'].value_counts())
# print(train['sorrow'].value_counts())
train.to_csv('huigui_ernie_fix_2.tsv', sep='\t', index=False)
