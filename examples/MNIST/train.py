import torch
from torch import nn
from torch.nn import functional as F

from opt import get_opts

from einops import rearrange, reduce, repeat

# datasets
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

# models
# from models.networks import LinearModel

# optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

seed_everything(1234, workers=True)


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MNISTSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()  # 继承所有LightningModule已有的属性
        self.save_hyperparameters(hparams)  # 记录每次训练的超参数设定
        # self.net = LinearModel(self.hparams.hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(28*28, self.hparams.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hparams.hidden_dim, 10)
            )
        
    def forward(self, x):
        """
        x: (B, 28, 28) batch of images
        """
#        x = x.flatten()  # (B, 28*28)
        x = rearrange(x, "b x y -> b (x y)") # or "b x y -> b (x y)", x=28, y=28
        return self.net(x)
        
    def prepare_data(self):
        """
        download data once
        """
        MNIST(self.hparams.root_dir, train=True, download=True)
        MNIST(self.hparams.root_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        dataset = MNIST(self.hparams.root_dir,
                        train=True,
                        download=False,
                        transform=T.ToTensor())
        train_length = len(dataset) # 60000
        self.train_dataset, self.val_dataset = \
            random_split(dataset,
                         [train_length-self.hparams.val_size, self.hparams.val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.hparams.num_epochs,
                                      eta_min=self.hparams.lr/1e2)   # 最小学习率

        return [self.optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits_predicted = self(images)

        loss = F.cross_entropy(logits_predicted, labels)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits_predicted = self(images)

        loss = F.cross_entropy(logits_predicted, labels)
        # accuracy: torchmetrics
        acc = torch.sum(torch.eq(torch.argmax(logits_predicted, -1), labels).to(torch.float32)) / len(labels) 

        log = {'val_loss': loss,
               'val_acc': acc}

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/acc', mean_acc, prog_bar=True)


if __name__ == '__main__':
    hparams = get_opts() # get options
    mnistsystem = MNISTSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',  # exp_name 指定实验名称
                              filename='{epoch:d}',
                              monitor='val/loss',
                              mode='max',
                              save_top_k=5)
    pbar = TQDMProgressBar(refresh_rate=1)
    callbacks = [ckpt_cb, pbar]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=callbacks,  # 存储、进度条
                      #resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      enable_model_summary=True, # 显示模型结构
                      accelerator='auto',         #指定用CPU还是GPU, auto会根据机器情况自动选择
                      devices=1,  # hparams.num_gous 指定GPU个数
                      num_sanity_val_steps=1, # 在开始第一步先执行val过程，防止后面val再报错就浪费时间了
                      profiler="simple" if hparams.num_gpus==1 else None,  # 显示模型瓶颈报告
                      #strategy=  # 分布式计算
                      benchmark=True) # cudnn: 加快速度，设定每次输入大小一致时比较实用
    trainer.fit(mnistsystem)