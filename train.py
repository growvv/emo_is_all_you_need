import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import time
import ipdb

from roledataset import RoleDataset, create_dataloader
from model import EmotionClassifier
from utils import load_checkpoint, save_checkpoint
from predict import predict
import config


#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'

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

            loss_love = criterion(outputs['love'], sample['love'].to(config.device))
            loss_joy = criterion(outputs['joy'], sample['joy'].to(config.device))
            loss_fright = criterion(outputs['fright'], sample['fright'].to(config.device))
            loss_anger = criterion(outputs['anger'], sample['anger'].to(config.device))
            loss_fear = criterion(outputs['fear'], sample['fear'].to(config.device))
            loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'].to(config.device))
            loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

            losses.append(loss.item())

            loss.backward()

#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            if global_step % log_steps == 0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train), 
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
    test_pred = predict(model, valid_loader)
    print(test_pred)    


    # ipdb.set_trace()
    submit = pd.read_csv('data/submit_example.tsv', sep='\t')

    label_preds = []
    for col in config.target_cols:
        preds = test_pred[col]
        label_preds.append(preds)
    print(len(label_preds[0]))
    sub = submit.copy()
    sub['emotion'] = np.stack(label_preds, axis=1).tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv('baseline_chinese-roberta-wwm-ext.tsv', sep='\t', index=False)
    sub.head()

do_train(model, train_loader, criterion, optimizer, scheduler)
