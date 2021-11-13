
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
    zero = 0
    one = 0
    two = 0
    three = 0
    four = 0
    five = 0
    six = 0
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
        if tmp == 0:
            zero += 1
        elif tmp == 1:
            one += 1
        elif tmp == 2:
            two += 1
        elif tmp == 3:
            three += 1
        elif tmp == 4:
            four += 1
        elif tmp == 5:
            five += 1
        elif tmp == 6:
            six += 1
        else:
            print(line[3])


    print("not zero: ", not_zero/full)
    print("all_zero: ", all_zero/full)
    print("zero: ", zero/full)
    print("one: ", one/full)
    print("two: ", two/full)
    print("three: ", three/full)
    print("four: ", four/full)
    print("five: ", five/full)
    print("six: ", six/full)
