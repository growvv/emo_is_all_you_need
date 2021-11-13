import torch

EPOCH_NUM=3
weight_decay=0.005
data_path='data'
warmup_proportion=0.0
batch_size=32
lr = 2e-5
max_len = 128

warm_up_ratio = 0
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']

# model
# roberta
#PRE_TRAINED_MODEL_NAME='models/chinese-roberta-wwm-ext'
#PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
# PRE_TRAINED_MODEL_NAME = '/home/liufarong/sdb1/PreModel/nghuyong_ernie-1.0'
# PRE_TRAINED_MODEL_NAME = '/home/liufarong/sdb1/PreModel/hfl_chinese-roberta-wwm-ext-large'
PRE_TRAINED_MODEL_NAME = '/home/liufarong/sdb1/PreModel/hfl_chinese-roberta-wwm-ext'

# result
res_tsv = "bert_gat.tsv"
run_plot = "runs/bert_gat"

# save model
model_root = "weights/checkpoint_bertgat.pth.tar"
load_model = False
save_model = False

adv_train = False
