from tqdm import tqdm
import pandas as pd
import ipdb

with open('data/train_dataset_v2.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    print("len: ", len(lines))
    data = list()
    lack = 0
    full = 0
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        if sp[0] == '' or sp[1] == '' or sp[2] == '' or sp[3] == '':
            #print("Error: ", sp)
            lack += 1
            continue
        data.append(sp)

    full = len(data)
    print("full: ", full)
    print("lack: ", lack)


    all_zero = 0
    not_zero = 0
    res = [0 for _ in range(6)]
    for line in data:
        #ipdb.set_trace()
        emos = list(map(int, line[3].split(',')))
        if line[3] == '0,0,0,0,0,0':
            all_zero += 1
        else :
            not_zero += 1

        tmp = 0
        for emo in emos:
            if emo > 0:
                tmp += 1
        res[tmp] += 1


    print("not zero: ", not_zero/full)
    print("all_zero: ", all_zero/full)
    for i in range(6):
        print("%d: %.2f" % (i, res[i]/full))

    for i in range(6):
        print("%d: %d" % (i, res[i]))