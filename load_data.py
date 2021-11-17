from tqdm import tqdm 
import pandas as pd
import ipdb
import numpy as np
from utils import seed_everything
from cleardata import clear_data
import re


seed_everything(19260817)

def load_train_dataset():
    with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
        lines = handler.read().split('\n')[1:-1]

        print('train before: ', len(lines))
        data = list()
        cnt0 = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        for line in tqdm(lines):
            #ipdb.set_trace()
        
            sp = line.split('\t')
            #if sp[0] == '34911_0044_A_660':
            #    ipdb.set_trace()
            if sp[0] == '' or sp[1] == '' or sp[2] == '' or sp[3] == '':
                #print("Error: ", sp)
                cnt0 += 1
                continue
            emos = list(map(int, sp[3].split(',')))
            if emos[0] == 0 and emos[1] == 0 and emos[2] == 0 and emos[3] == 0 and emos[4] == 0 and emos[5] == 0:
                if np.random.rand(1)[0] < 0.0:
                    continue                                 
            if len(sp[1]) > 120:  # 过长不要
                cnt1 += 1
                continue       
 
            if re.search(sp[2], sp[1]) == None:  # 标注出错,角色名并不在句子中
                #print(sp)
                cnt2 += 1
                continue

            sp[1] = clear_data(sp[1])   # 给角色名前后增加空格        
            cnt3 += 1
            data.append(sp)

    print('train after: ', cnt0, cnt1, cnt2, cnt3, len(data))

    train = pd.DataFrame(data)
    train.columns = ['id', 'content', 'character', 'emotions']

    train = train[train['emotions'] != '']
    train['emotions'] = train['emotions'].apply(lambda x: [int(_i) for _i in x.split(',')]) 
    train[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = train['emotions'].values.tolist()
    train['text'] = train['content'].astype(str)

    print(train.text.head(5))
    train.to_csv('data/train.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'], sep='\t', index=False)


def load_test_dataset():
    no_emo_ids = []
    data = []
    with open('data/test_dataset.tsv', 'r', encoding='utf-8') as handler:
        lines = handler.read().split('\n')[1:-1]

        print('test before: ', len(lines))
        data = list()
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        for line in tqdm(lines):
            #ipdb.set_trace()
            sp = line.split('\t')
            if sp[0] == '' or sp[1] == '' or sp[2] == '': # 缺失的记录下来,直接填全0
                no_emo_ids.append(sp[0])
                sp[1] = 'a1正在吃饭'
                sp[2] = 'a1'  # 填充默认值
                #continue

            sp[1] = clear_data(sp[1])   # 给角色名前后增加空格
            
            if re.search(sp[2], sp[1]):  # 截取含有角色名部分
                start,end = re.search(sp[2], sp[1]).span()
                if start > 120:
                    sp[1] = sp[1][max(0, start-64): start+64]

            data.append(sp)

    test = pd.DataFrame(data)
    test.columns = ['id', 'content', 'character']
    #test = pd.read_csv('data/test_dataset.tsv', sep='\t')  

    #test['content'] = [clear_data(text) for text in test['content']]
    test['text'] =  test['content'].astype(str)
    test[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] =[0,0,0,0,0,0]

    print('test after: ', len(test))
    print(test.text.head(5))

    test.to_csv('data/test.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],                 sep='\t',
                 index=False)
    return no_emo_ids

def get_lack():
    no_emo_ids = []
    with open('data/test_dataset.tsv', 'r', encoding='utf-8') as handler:
        lines = handler.read().split('\n')[1:-1]

        for line in lines:
            #ipdb.set_trace()
            sp = line.split('\t')
            if sp[0] == '' or sp[1] == '' or sp[2] == '': # 缺失的记录下来,直接填全0
                no_emo_ids.append(sp[0])
            if sp[0] == '':
                print('不会吧!?')

    return no_emo_ids


if __name__ == '__main__':
    #load_train_dataset()
    #no_emo_ids = load_test_dataset()
    no_emo_ids = get_lack()
    print(len(no_emo_ids))
    print(no_emo_ids)
    print('finish!!')
