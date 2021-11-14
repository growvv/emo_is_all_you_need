from tqdm import tqdm 
import pandas as pd
import ipdb
import numpy as np
from utils import seed_everything
#from cleardata import clear_data
import re


seed_everything(19260817)

def myreplace(matched):
    matched = matched.group()
    return ' ' +  matched + ' '

def clear_data(str):
    str = re.sub('[a-z][0-9]', myreplace, str, flags=re.I)
    return str


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
        if len(sp[1]) > 125:
            continue       
 
        if re.search(sp[2], sp[1]) == None:  # 标注出错,角色名并不在句子中
            print(sp)
            continue

        clear_data(sp[1])   # 给角色名前后增加空格        

        data.append(sp)

print(len(data))

train = pd.DataFrame(data)
train.columns = ['id', 'content', 'character', 'emotions']

test = pd.read_csv('data/test_dataset.tsv', sep='\t')
submit = pd.read_csv('data/submit_example.tsv', sep='\t')
train = train[train['emotions'] != '']

#ipdb.set_trace()
#train['text'] = [clear_data(text) for text in train['content'].astype(str)]
#test['text'] = [clear_data(text) for text in test['content'].astype(str)]
train['text'] = clear_data(train['content'].astype(str)) 
test['text'] =  clear_data(test['content'].astype(str))

train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] =[0,0,0,0,0,0]

print(train.text.head(5))
print(test.text.head(5))

train.to_csv('data/train.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)

test.to_csv('data/test.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)
