import torch
import torch.nn as nn
from torch.utils.tensorboard.summary import text
from tqdm import tqdm
from collections import defaultdict
import config
from rmseloss import RMSELoss
import ipdb
import pandas as pd
import numpy as np

rmseloss = nn.MSELoss()

def validate(model, validate_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    for step, batch in tqdm(enumerate(validate_loader)):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        text = batch['text']
        character = batch['character']
        # target = batch
        with torch.no_grad():
            logists = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character)
            val_loss += rmseloss(logists, batch['labels'].to(config.device))

    return val_loss / len(validate_loader)


def predict(model, test_loader):
    model.eval()
    label_preds = None
    for step, batch in tqdm(enumerate(test_loader)):
        input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        text = batch['text']
        character = batch['character']
        with torch.no_grad():
            logists = model(input_ids=input_ids, attention_mask=attention_mask, text=text, character=character)
            if label_preds is None:
                label_preds = logists
            else:
                label_preds = torch.cat((label_preds, logists), dim=0)

    # ipdb.set_trace()
    sub = pd.read_csv('data/submit_example.tsv', sep='\t')

    print(len(sub['emotion']))
    sub['emotion'] = label_preds.tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv(config.res_tsv, sep='\t', index=False)
    print(sub.head(5))
