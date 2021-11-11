from tqdm import tqdm
import pandas as pd
import ipdb
import numpy as np


with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    print(len(lines))
    data = list()
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        if sp[0] == '' or sp[1] == '' or sp[2] == '' or sp[3] == '':
            # print("Error: ", sp)
            continue
        emos = list(map(int, sp[3].split(',')))
        if emos[0] == 0 and emos[1] == 0 and emos[2] == 0 and emos[3] == 0 and emos[4] == 0 and emos[5] == 0:
            if np.random.rand(1)[0] < 0.3:
                continue
        
        data.append(sp)

print(len(data))