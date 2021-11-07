import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
import config
from utils import load_checkpoint
from transformers import AdamW, get_linear_schedule_with_warmup
from model import EmotionClassifier
import ipdb
import pandas as pd
import numpy as np


rmseloss = nn.MSELoss()

def validate(model, validate_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    for step, batch in tqdm(enumerate(validate_loader)):
        b_input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            target = None
            #ipdb.set_trace()
            for col in config.target_cols:
                if target == None:
                    target = batch[col].unsqueeze(1).to(config.device)
                else:
                    target = torch.cat((target, batch[col].unsqueeze(1).to(config.device)), dim=1)
            out = torch.argmax(logists, axis=2)
            val_loss = rmseloss(out, target)

    return val_loss / len(validate_loader)


def predict(model, test_loader):
    label_preds = None
    model.eval()
    for step, batch in tqdm(enumerate(test_loader)):
        b_input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            out = torch.argmax(logists, axis=2)  # (batch_size, 6)
            #val_loss = rmseloss(out2, batch[col].to(config.device))
            #ipdb.set_trace()
            if label_preds is None:
                label_preds = out
            else:
                label_preds = torch.cat((label_preds, out), dim=0)

    sub = pd.read_csv('data/submit_example.tsv', sep='\t')

    #ipdb.set_trace()
    print(len(sub['emotion']))
    sub['emotion'] = label_preds.tolist()
    sub['emotion'] = sub['emotion'].apply(lambda x: ','.join([str(i) for i in x]))
    sub.to_csv(config.res_tsv, sep='\t', index=False)
    print(sub.head(5))


if __name__ == "__main__":
    model = EmotionClassifier(n_classes=4, bert=base_model).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    load_checkpoint(torch.load(config.model_root), model, optimizer)
