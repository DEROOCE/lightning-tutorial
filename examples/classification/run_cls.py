import os
import copy
import torch
import json
import random
from tqdm import tqdm
import numpy as np
from torch import nn
from config import set_args
from model import Model
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers.models.roformer import RoFormerTokenizer
from data_helper import load_data, CustomDataset, collate_fn
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

#def set_seed():
#    os.environ['PYTHONHASHSEED'] = str(args.seed)
#    random.seed(args.seed)
#    np.random.seed(args.seed)
#    torch.manual_seed(args.seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed(args.seed)
#        torch.cuda.manual_seed_all(args.seed)
#    
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    args = set_args(Trainer)
    #set_seed()
    seed_everything(3407, workers=True)
    
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = RoFormerTokenizer.from_pretrained(args.pretrained_model_path)
    label_num = 119
    model = Model(args, label_num, tokenizer)

    print("***** Running training *****")
    ckpt_callbacks = ModelCheckpoint(dirpath=f'ckpt/cls',
                                    filename='{epoch:d}',
                                    monitor='val/loss',
                                    mode='min',
                                    save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_callbacks, pbar]
    
    logger = TensorBoardLogger(save_dir='logs',
                                name = 'roformer-cls',
                                default_hp_metric=False)
    trainer = Trainer(max_epochs=args.num_train_epochs,
                    callbacks=callbacks,
                    logger=logger,
                    enable_model_summary=True,
                    accumulate_grad_batches=args.gradient_accumulation_steps,
                    accelerator='auto', devices=1,
                    num_sanity_val_steps=1,
                    profiler='simple' if args.num_gpus==1 else None)
    trainer.fit(model)
    
    