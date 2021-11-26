from load_data import get_lack
import pandas as pd
import ipdb


# /results/bert_gat_adv.tsv
def ensemble(files, weights):
    results = []
    for file in files:
        results.append(pd.read_csv(file, sep='\t'))

    for result in results:
        print(result.head())

    #ipdb.set_trace()
    for result in results:
        result['emotions'] = result['emotion'].apply(lambda x: [_i for _i in x.split(',')])
        result[['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']] = result['emotions'].values.tolist()

    ret = pd.DataFrame(columns=['id', 'emotion','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'], index=range(len(results[0])))
    ret['id'] = results[0]['id']
    ret['emotion'] = ''
    emos = ['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']
    for emo in emos:
        ret[emo] = 0
        for i in range(len(results)):
            #ipdb.set_trace()
            results[i][emo] = results[i][emo].map(lambda x: float(x)*weights[i])
            ret[emo] += results[i][emo].values.tolist()
        #results[i]['love'].map(lambda x: float(x)*weights[i])
    print(ret.head())    
   
    ret['emotion'] = ret['love'].map(str) + ',' + ret['joy'].map(str) + ',' + ret['fright'].map(str) + ',' + ret['anger'].map(str) + ',' + ret['fear'].map(str) + ',' + ret['sorrow'].map(str)

    #ipdb.set_trace()
    ret.drop(['love', 'joy', 'fright', 'anger', 'fear', 'sorrow'], axis=1, inplace=True)
    ret = ret.reset_index(drop=True)
    ret.to_csv('./results/bert_gat_ensemble.tsv',columns=['id',  'emotion'], sep='\t', index=False)

    print(ret.head())

if __name__ == "__main__":
    #ensemble(['./results/bert_gat_adv.tsv'], [1.0])
    ensemble(['./results/bert_gat_adv_improve.tsv', './results/bert_gat_adv_gat11_improve.tsv', './results/bert_gat_normal_improve.tsv'], [1/3, 1/3, 1/3])
