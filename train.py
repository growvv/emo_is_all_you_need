import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
import time
import ipdb

from roledataset import RoleDataset, create_dataloader
from model import EmotionClassifier
from utils import load_checkpoint, save_checkpoint
from predict import predict
import config


# roberta
PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
base_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
# model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

trainset = RoleDataset(tokenizer, config.max_len, mode='train')
train_loader = create_dataloader(trainset, config.batch_size, mode='train')
valset = RoleDataset(tokenizer, config.max_len, mode='test')
valid_loader = create_dataloader(valset, config.batch_size, mode='test')

model = EmotionClassifier(n_classes=4, bert=base_model).to(config.device)
optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
if config.load_model:
    load_checkpoint(torch.load(config.model_root), model, optimizer)

total_steps = len(train_loader) * config.EPOCH_NUM

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps = config.warm_up_ratio * total_steps,
  num_training_steps = total_steps
)

criterion = nn.CrossEntropyLoss()

writer = SummaryWriter("runs/loss_plot")

def do_train(model, date_loader, criterion, optimizer, scheduler, metric=None):
    model.train()
    tic_train = time.time()
    log_steps = 1
    global_step = 0
    for epoch in range(config.EPOCH_NUM):
        losses = []
        for step, sample in enumerate(train_loader):
            input_ids = sample["input_ids"].to(config.device)
            attention_mask = sample["attention_mask"].to(config.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) 

            # target = torch.empty(64, 0)

            love = sample['love'].to(config.device).unsqueeze(1)
            joy = sample['joy'].to(config.device).unsqueeze(1)
            fright = sample['fright'].to(config.device).unsqueeze(1)
            anger = sample['anger'].to(config.device).unsqueeze(1)
            fear = sample['fear'].to(config.device).unsqueeze(1)
            sorrow = sample['sorrow'].to(config.device).unsqueeze(1)

            target = torch.cat((love, joy), 1)
            target = torch.cat((target, fright), 1)
            target = torch.cat((target, anger), 1)
            target = torch.cat((target, fear), 1)
            target = torch.cat((target, sorrow), 1)
            # ipdb.set_trace()

            outputs = outputs.reshape(-1, 4)
            target = target.reshape(-1)
            loss = criterion(outputs, target)  # [64, 6, 4], [64, 6]

            losses.append(loss.item())

            loss.backward()

#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, loss, global_step / (time.time() - tic_train), 
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

    # 评估
    model.eval()
    predict(model, valid_loader)


do_train(model, train_loader, criterion, optimizer, scheduler)