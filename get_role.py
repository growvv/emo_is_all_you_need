import pandas as pd
import ipdb
import re

def get_role():
    with open('data/train.csv', 'r', encoding='utf-8') as f:
        # ipdb.set_trace()
        lines = f.read().split('\n')
        lines = [line.split('\t') for line in lines[1:]]
        # ipdb.set_trace()
        data = pd.DataFrame(lines)

        role = data[2].to_list()

        role = list(set(role))

        return role


if __name__ == '__main__':
    roles = get_role()
    str = '天空下着暴雨，o2正在给c1穿雨衣，他自己却只穿着单薄的军装。'
    print(roles)
    print(len(roles))

    exist_roles = [role for role in roles if re.search(role, str)]

    print(exist_roles)
    print(len(exist_roles))
