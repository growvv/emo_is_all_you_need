import ipdb


with open('data/train.csv', 'r', encoding='utf-8') as f:
    # ipdb.set_trace()
    lines = f.read().split('\n')
    lines = [line.split('\t')[0] for line in lines[1:]]

    pre_scene = lines[0].split('_')[0] + '_' +  lines[0].split('_')[1]
    start = 0
    batch_sizes = []
    d = {}
    # ipdb.set_trace()
    for i in range(len(lines)):
        # ipdb.set_trace()
        # print(lines[i])
        drama, scene, _, _ = lines[i].split('_')
        scene = drama + '_' + scene
        if scene != pre_scene:
            pre_scene = scene
            batch_sizes.append((start, i))
            start = i

        if scene not in d:
            d[scene] = 1
        else:
            d[scene] += 1

    batch_sizes.append((start, len(lines)))
    print(len(batch_sizes))
    step = 0
    for k, v in sorted(d.items(), key=lambda x: x[1], reverse=True):
        # print(k, v)
        if v > 16:
            step += 1
    print(len(d))
    print(step/len(d))
    # ipdb.set_trace()



