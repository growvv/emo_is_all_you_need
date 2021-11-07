import torch
import numpy as np
import random


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

<<<<<<< HEAD
=======

>>>>>>> 8924bc6 (fix bug)
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
<<<<<<< HEAD
=======



if __name__ == "__main__":
    seed_everything(seed=19260817)
    print(random.random())
    print(np.random.random())
    print(torch.rand(1))
    print(torch.cuda.is_available())
>>>>>>> 8924bc6 (fix bug)
