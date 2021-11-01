import torch

EPOCH_NUM=3
weight_decay=0.005
data_path='data'
warmup_proportion=0.0
batch_size=64
lr = 2e-5
max_len = 128

warm_up_ratio = 0
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_cols=['love', 'joy', 'fright', 'anger', 'fear', 'sorrow']

# model
# roberta
PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'

# save model
model_root = "my_checkpoint2.pth.tar"
load_model = False
save_model = True
