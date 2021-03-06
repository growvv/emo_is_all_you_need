#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset

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

seed_everything(seed=19260817)

# robert0
tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # load pretain model

all_dataset = RoleDataset(tokenizer, config.max_len, mode='train')
#all_dataset = Subset(all_dataset, range(config.batch_size*1000, len(all_dataset)))  # 用于小规模调试
train_size = int(len(all_dataset) * 0.9)  # use 90% data for train, 10% for validate
validate_size = len(all_dataset) - train_size
train_dataset = torch.utils.data.Subset(all_dataset, range(train_size)) 
validate_dataset = torch.utils.data.Subset(all_dataset, range(train_size, len(all_dataset)))
# train_dataset, validate_dataset = torch.utils.data.random_split(trainset, [train_size, validate_size])  # because we need data is in order, random can`t meet need
train_loader = create_dataloader(train_dataset, config.batch_size, shuffle=False)
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

def do_train(model, criterion, optimizer, scheduler, metric=None):
    model.train()
    tic_train = time.time()
    log_steps = 1
    global_step = 0
    for epoch in range(config.EPOCH_NUM):
        losses = []
        losses_adv = []
    
        for step, sample in enumerate(train_loader):
            if config.debug:
                if step == 3:
                    break
            input_ids = sample["input_ids"].to(config.device)
            attention_mask = sample["attention_mask"].to(config.device)
            text = sample["text"]
            character = sample["character"]
            id = sample["id"]
            pos = sample["pos"].to(config.device)
            labels = sample["labels"].to(config.device)
            # print(id, text)
            #ipdb.set_trace()

            # 1. 正常训练
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character, pos=pos)

            loss = criterion(outputs, sample["labels"].to(config.device))

            losses.append(loss.item())

            loss.backward()

            # 2. 加入对抗训练
            loss_adv = 0
            if config.adv_train:
                fgm.attack(epsilon=0.3, emb_name="word_embeddings") # 只攻击word embedding
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character, pos=pos)
                loss_adv = criterion(outputs, sample["labels"].to(config.device))
                losses_adv.append(loss_adv.item())
                loss_adv.backward()
                fgm.restore(emb_name="word_embeddings") # 恢复Embedding的参数

            # 梯度下降，更新参数
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % log_steps == 0:
               print("global step %d, epoch: %d, batch: %d, loss: %.5f, loss_adv: %.5f, speed: %.2f step/s, lr: %.10f" % (global_step, epoch, step, loss, loss_adv, global_step / (time.time() - tic_train), 
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
