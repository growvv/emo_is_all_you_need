from tqdm import tqdm
import pandas as pd
import ipdb

with open('huigui_ernie_fix.tsv', 'r', encoding='utf-8') as handler:
    lines = handler.read().split('\n')[1:-1]

    print("len: ", len(lines))
    data = list()
    lack = 0
    full = 0
    for line in tqdm(lines):
        # ipdb.set_trace()
        sp = line.split('\t')
        data.append(sp[1])

    full = len(data)
    print("full: ", full)
    print("lack: ", lack)


    all_zero = 0
    not_zero = 0
    res = [0 for _ in range(7)]
    for line in data:
        #ipdb.set_trace()
        # print(line)
        emos = list(map(float, line.split(',')))
        # print(emos)

        tmp = 0
        for emo in emos:
            if emo > 0.0:
                tmp += 1
        res[tmp] += 1


    print("not zero: ", not_zero/full)
    print("all_zero: ", all_zero/full)
    for i in range(7):
        print("%d: %.2f" % (i, res[i]/full))

    for i in range(7):
        print("%d: %d" % (i, res[i]))