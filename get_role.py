import pandas as pd
import ipdb
import re

import argparse

def get_role_part(path):
    with open(path, 'r', encoding='utf-8') as f:
        #ipdb.set_trace()
        lines = f.read().split('\n')
        lines = [line.split('\t') for line in lines[1:]]
        # ipdb.set_trace()
        data = pd.DataFrame(lines)

        roles = data[2].to_list()[:-1]  # 去掉最后一个空行，为啥会这样??

        roles = list(set(roles))

    return roles

def get_role():
    roles1 = get_role_part('data/train_dataset_v2.tsv')
    roles2 = get_role_part('data/test_dataset.tsv')
    roles1.extend(roles2)
    roles = list(set(roles1))
    roles = [role for role in roles if role != '']
    return roles


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='Process dataset path')
 
    #parser.add_argument('--test_dataset', dest="test_path" ,type=str, help='input file path')
    #parser.add_argument('--train_dataset', dest="train_path" ,type=str, help='input file path')

    #args = parser.parse_args()

    #roles1 = get_role(args.test_path, args.train_path)
    
    roles = get_role()
    roles = list(set(roles))
    print(roles)

    str = '天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装。'
    exist_roles = [role for role in roles if re.search(role, str)]

    print(exist_roles)
    print(len(exist_roles))
