import torch
from tqdm import tqdm
from collections import defaultdict
import config
from utils import load_checkpoint
from transformers import AdamW, get_linear_schedule_with_warmup
from model import EmotionClassifier


def predict(model, test_loader):
    val_loss = 0
    test_pred = defaultdict(list)
    model.eval()
    for step, batch in tqdm(enumerate(test_loader)):
        b_input_ids = batch['input_ids'].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        with torch.no_grad():
            logists = model(input_ids=b_input_ids, attention_mask=attention_mask)
            for col in config.target_cols:
                out2 = torch.argmax(logists[col], axis=1)
                test_pred[col].extend(out2.cpu().numpy().tolist())

    return test_pred

if __name__ == "__main__":
    model = EmotionClassifier(n_classes=4, bert=base_model).to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    load_checkpoint(torch.load(config.model_root), model, optimizer)
