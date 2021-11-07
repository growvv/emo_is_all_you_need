import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import math

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
import time
import ipdb

from roledataset import RoleDataset, create_dataloader
from model import EmotionClassifier
from utils import load_checkpoint, save_checkpoint
from predict import predict, validate
import config


# roberta
#PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
base_model = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)  # 加载预训练模型
# model = ppnlp.transformers.BertForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=2)

trainset = RoleDataset(tokenizer, config.max_len, mode='train')

train_size = int(len(trainset) * 0.95)
validate_size = len(trainset) - train_size

train_dataset, validate_dataset = torch.utils.data.random_split(trainset, [train_size, validate_size])
train_loader = create_dataloader(train_dataset, config.batch_size, mode='train')
validate_loader = create_dataloader(validate_dataset, config.batch_size, mode='train')

#train_loader = create_dataloader(trainset, config.batch_size, mode='train')
testset = RoleDataset(tokenizer, config.max_len, mode='test')
test_loader = create_dataloader(testset, config.batch_size, mode='test')

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

criterion = nn.MSELoss()

writer = SummaryWriter(config.run_plot)

def do_train(model, date_loader, criterion, optimizer, scheduler, metric=None):
    model.train()
    tic_train = time.time()
    log_steps = 1
    global_step = 0
    for epoch in range(config.EPOCH_NUM):
        losses = []
        for step, sample in enumerate(train_loader):
            if step == 3:
                break
            input_ids = sample["input_ids"].to(config.device)
            attention_mask = sample["attention_mask"].to(config.device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask) 
            # outputs: [64, 6]
            # target = torch.empty(64, 0)

            target = None
            #ipdb.set_trace()
            for col in config.target_cols:
                if target == None:
                    target = sample[col].unsqueeze(1).to(config.device)
                else:
                    target = torch.cat((target, sample[col].unsqueeze(1).to(config.device)), dim=1)
         
            #ipdb.set_trace() 
            outputs = torch.argmax(outputs, axis=2) # [64, 6, 4] ->  [64, 6]
            loss = criterion(outputs, target).requires_grad_(True)

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

        # 验证
        model.eval()
        validate_pred = validate(model, validate_loader)
        print("score: %f" % (validate_pred))
        print("score: %f" % (1/(1+math.sqrt(validate_pred))))

    #评估
    model.eval()
    predict(model, test_loader)


do_train(model, train_loader, criterion, optimizer, scheduler)
