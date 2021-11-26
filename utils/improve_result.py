from load_data import get_lack
import pandas as pd
import ipdb


lack_ids = get_lack()
print(lack_ids)
#ipdb.set_trace()

# ./results/bert_gat_adv.tsv
def improve(file):
    f = open(file, 'r', encoding='utf-8')

    lines = f.read().split('\n')[1:-1]

    data = []
    for line in lines:
        sp = line.split('\t')
        if sp[0] in lack_ids:
            sp[1] = '0.0, 0.0, 0.0, 0.0, 0.0, 0.0'
        data.append(sp)    

    res = pd.DataFrame(data)
    res.columns = ['id', 'emotion']

    print(res.head(5))
    #ipdb.set_trace()
    file = file.split('.')
    path = '.' + file[0] + file[1] + '_improve' + '.tsv' 
    print(path)
    res.to_csv(path, sep='\t', index=False)


if __name__ == "__main__":
    improve('./results/bert_gat_adv.tsv')
    improve('./results/bert_gat_adv_gat11.tsv')
    improve('./results/bert_gat_normal.tsv')



