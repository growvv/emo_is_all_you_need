from tqdm import tqdm 
import pandas as pd
import ipdb
from transformers import BertTokenizer
import seaborn as sns
import matplotlib.pyplot as plt 


with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    data = list()
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        if sp[0] == '' or sp[1] == '' or sp[2] == '' or sp[3] == '':
            # print("Error: ", sp)
            continue
        data.append(sp)

train = pd.DataFrame(data)
ipdb.set_trace()
train.columns = ['id', 'content', 'character', 'emotions']

print(train.head())

# train = train[train['emotions'] != '']
print(len(train))
print(train.shape)

train['text'] = train[ 'content'].astype(str)  +'角色: ' + train['character'].astype(str)

train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')])
print(train.head())

train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
print(train.head())

print(train['love'].value_counts())
print(train['joy'].value_counts())
print(train['fright'].value_counts())
print(train['anger'].value_counts())
print(train['fear'].value_counts())
print(train['sorrow'].value_counts())
# print(train['id'].value_counts())
# ipdb.set_trace()

d = {}
d2 = {}
d3 = {}
for  i in range(len(train)):
    # ipdb.set_trace()
    item, character = train['id'][i], train['character'][i]
    drama, scene, _, _ = item.split('_')
    if drama not in d:
        d[drama] = 1
    else:
        d[drama] += 1

    x = drama + '_' + scene
    if x not in d2:
        d2[x] = 1
    else:
        d2[x] += 1

    x = drama + '_' + character
    if x not in d3:
        d3[x] = 1
    else:
        d3[x] += 1


    

print(len(d))
print(len(d2))
print(len(d3))
ipdb.set_trace()
#31
#3147
#575
# print(sorted(d.items(), key = lambda kv:(kv[1], kv[0]))[:10]) 
# print(sorted(d2.items(), key = lambda kv:(kv[1], kv[0])).reverse()[:10]) 

# for k, v in d.items():
#     print(k, v)
