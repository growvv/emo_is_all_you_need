from tqdm import tqdm 
import pandas as pd
import ipdb
import numpy as np
from utils import seed_everything

seed_everything(19260817)

with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    print(len(lines))
    data = list()
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        if sp[0] == '' or sp[1] == '' or sp[2] == '' or sp[3] == '':
            #print("Error: ", sp)
            continue
        emos = list(map(int, sp[3].split(',')))
        if emos[0] == 0 and emos[1] == 0 and emos[2] == 0 and emos[3] == 0 and emos[4] == 0 and emos[5] == 0:
            if np.random.rand(1)[0] < 0.0:
                continue                                 
        
        data.append(sp)

print(len(data))

train = pd.DataFrame(data)
train.columns = ['id', 'content', 'character', 'emotions']

test = pd.read_csv('data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('data/submit_example.tsv', sep='\t')
train = train[train['emotions'] != '']


train['text'] = train['content'].astype(str) 
test['text'] =  test['content'].astype(str)

train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] =[0,0,0,0,0,0]

print(train.text.head())
print(test.text.head())

train.to_csv('data/train.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)

test.to_csv('data/test.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)
