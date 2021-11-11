import ipdb
import torch
import random
import numpy as np
import ipdb
import pandas as pd
from torch.functional import norm
from transformers.tokenization_utils_base import BatchEncoding


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_batch_sizes(file_path):
    with open(file_path, "r") as f:
        lines =  [line.strip('\n').split('\t') for line in f.readlines()]

    train = pd.DataFrame(lines[1:], columns=['id', 'content', 'character', 'text', 'love', 'joy', 'fright', 'anger', 'fear','sorrow'])
    # ipdb.set_trace()
    batch_sizes = []
    pre_drama = train['id'][0].split('_')[:2]    
    start = 0
    for index in range(len(train['id'])):
        drama = train['id'][index].split('_')[:2]
        # print(drama)
        if drama != pre_drama:
            batch_sizes.append((start, index))     # [start, end)
            pre_drama = drama
            start = index

    batch_sizes.append((start, len(train['id'])))
              


    return batch_sizes


if __name__ == "__main__":
    batch_sizes = get_batch_sizes("./data/train.csv")
    print(len(batch_sizes))
    mymax = 0
    mys = 0
    mye = 0
    d = {}
    for s, e in batch_sizes:
        # print(s, e, e-s)
        if e-s > mymax:
            mymax = e-s
            mys = s
            mye = e

        len = e-s
        if len not in d:
            d[len] = 1
        else:
            d[len] += 1

    # ipdb.set_trace()
    print(mymax, mys, mye)
    for k, v in sorted(d.items()):
        print(k, v)
    

    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-white')

    data = [ e-s for s, e in batch_sizes]
    #最基本的频次直方图命令
    plt.hist(data, bins=200, color='#0504aa', alpha=0.7)
    plt.xlim(0, 100)
    # plt.show()