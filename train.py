#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import math
import ipdb

from roledataset import RoleDataset, create_dataloader
from model import EmotionClassifier
from utils import load_checkpoint, save_checkpoint, seed_everything
from predict import predict, validate
import config

from fgm import FGM

from torch.utils.data import Subset

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
seed_everything(seed=19260817)

# roberta
#PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
#PRE_TRAINED_MODEL_NAME = 'nghuyong/ernie-1.0'
#PRE_TRAINED_MODEL_NAME = '/home/liufarong/sdb1/Test_Bert/hfl_chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
# model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

all_dataset = RoleDataset(tokenizer, config.max_len, mode='train')
train_size = int(len(all_dataset) * 0.95)
# validate_size = len(all_dataset) - train_size
# train_loader = create_dataloader(trainset, config.batch_size, mode='train')
train_dataset = torch.utils.data.Subset(all_dataset, range(train_size)) 
validate_dataset = torch.utils.data.Subset(all_dataset, range(train_size, len(all_dataset)))
# train_dataset, validate_dataset = torch.utils.data.random_split(trainset, [train_size, validate_size])
# train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=False)
validate_loader = create_dataloader(validate_dataset, config.batch_size, shuffle=False)

testset = RoleDataset(tokenizer, config.max_len, mode='test')
test_loader = create_dataloader(testset, config.batch_size, shuffle=False)

model = EmotionClassifier(n_classes=6, bert=base_model).to(config.device)
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
if config.load_model:
    load_checkpoint(torch.load(config.model_root, map_location=torch.device('cpu')), model, optimizer)

total_steps = len(train_dataset) * config.EPOCH_NUM

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps = config.warm_up_ratio * total_steps,
  num_training_steps = total_steps
)

criterion = nn.MSELoss()

writer = SummaryWriter(config.run_plot)

fgm = FGM(model)
# print(model)
# ipdb.set_trace()
def do_train(model, criterion, optimizer, scheduler, metric=None):
    model.train()
    tic_train = time.time()
    log_steps = 1
    global_step = 0
    for epoch in range(config.EPOCH_NUM):
        losses = []
        losses_adv = []
        for index in range(len(train_dataset)):
            # ipdb.set_trace()
            start = max(0, index-config.batch_size//2)
            end = min(len(train_dataset), index+config.batch_size//2)+1
            # batch_data = train_dataset[start:end]
            batch_data = Subset(train_dataset, range(start, end))
            offset = index - start
            train_loader = create_dataloader(batch_data, config.batch_size, shuffle=False)
    
            for step, sample in enumerate(train_loader):
                input_ids = sample["input_ids"].to(config.device)
                attention_mask = sample["attention_mask"].to(config.device)
                text = sample["text"]
                character = sample["character"]
                id = sample["id"]
                pos = sample["pos"]
                # print(id, text)
                # ipdb.set_trace()

                # 1. 正常训练
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character, pos=pos, offset=offset)

                # ipdb.set_trace()
                loss = criterion(outputs, sample["labels"][offset].to(config.device))

                losses.append(loss.item())

                loss.backward()

                # 2. 加入对抗训练
                loss_adv = 0
                if config.adv_train:
                    fgm.attack(epsilon=0.3, emb_name="word_embeddings") # 只攻击word embedding
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character)
                    loss_adv = criterion(outputs, sample["labels"][offset].to(config.device))
                    losses_adv.append(loss_adv.item())
                    loss_adv.backward()
                    fgm.restore(emb_name="word_embeddings") # 恢复Embedding的参数

                # 梯度下降，更新参数
    #             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                global_step += 1

                if global_step % log_steps == 0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.5f, loss_adv: %.5f, speed: %.2f step/s, lr: %.10f"
                        % (global_step, epoch, step, loss, loss_adv, global_step / (time.time() - tic_train), 
                            float(scheduler.get_last_lr()[0])))

                writer.add_scalar("Training loss", loss, global_step=global_step)

        # 每一轮epoch
        # save model
        if config.save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=config.model_root)

        # 验证
        model.eval()
        validate_pred = validate(model, validate_loader)
        print("score: %f" % (validate_pred))
        print("score: %f" % (1/(1+math.sqrt(validate_pred))))

    # 测试
    model.eval()
    test_pred = predict(model, test_loader)


do_train(model, criterion, optimizer, scheduler)
