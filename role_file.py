import ipdb
from get_role import get_role


roles = get_role()
print(roles)
esixt = []

with open('models/chinese-roberta-wwm-ext/vocab.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        # print(line)
        # ipdb.set_trace()

        if line in roles:
            esixt.append(line)
        

not_exist = []
for role in roles:
    if role not in esixt:
        not_exist.append(role)


for role in not_exist:
    print(role)

print(len(not_exist))
    


    
        