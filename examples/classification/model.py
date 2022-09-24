"""
@file   : model.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-22
"""
import json 
from sklearn import metrics
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_helper import CustomDataset, load_data, collate_fn
from transformers.models.roformer import RoFormerModel, RoFormerConfig
from transformers import AdamW, get_linear_schedule_with_warmup

class Classifier(nn.Module):
    # 加个全连接 进行分类
    def __init__(self, num_cls):
        super(Classifier, self).__init__()
        self.dense1 = torch.nn.Linear(768, 384)
        self.dense2 = torch.nn.Linear(384, num_cls)
        self.activation = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x

class Model(pl.LightningModule):
    def __init__(self, args, label_num, tokenizer):
        super(Model, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.config = RoFormerConfig.from_pretrained('./pretrain/config.json')
        self.config.output_hidden_states = True   # 输出所有的隐层
        self.config.output_attentions = True  # 输出所有注意力层计算结果
        self.roberta = RoFormerModel.from_pretrained('./pretrain', config=self.config)
        self.tokenizer = tokenizer
        self.loss_func = nn.CrossEntropyLoss()
        num_cls = label_num
        # self.highway = Highway(size=768, num_layers=3)
        self.classifier = Classifier(num_cls)
        self.train_label = []
        self.train_predict = []
        self.eval_targets = []
        self.eval_predict = []

    def forward(self, input_ids, attention_mask, segment_ids):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        # output[0].size(): batch_size, max_len, hidden_size
        # output[1].size(): batch_size, hidden_size
        # len(output[2]): 13, 其中一个元素的尺寸: batch_size, max_len, hidden_size
        # len(output[3]): 12, 其中一个元素的尺寸: batch_size, 12层, max_len, max_len
        # 这里采用最后一层所有输出的池化
        all_output, _ = torch.max(output[0], dim=1)
        logits = self.classifier(all_output)
        return logits
    
    def prepare_data(self):
        # 加载数据
        label2id = json.load(open(self.args.label2id_path, 'r', encoding='utf8'))
        self.train_df = load_data(self.args.train_data_path, label2id)
        self.val_df = load_data(self.args.dev_data_path, label2id)
        print('训练集的大小:', self.train_df.shape)
        print('验证集的大小:', self.val_df.shape)
        
    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        self.train_dataset = CustomDataset(dataframe=self.train_df, tokenizer=self.tokenizer)
        self.val_dataset = CustomDataset(dataframe=self.val_df, tokenizer=self.tokenizer)

    def train_dataloader(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, 
                                    shuffle=True,
                                    batch_size=self.args.train_batch_size,
                                    collate_fn=collate_fn,
                                    pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                        shuffle=False,
                        batch_size=self.args.val_batch_size,
                        collate_fn=collate_fn,
                        num_workers=4,
                        pin_memory=True)
    
    def calculate_loss(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, label_ids = batch
        logits = self(input_ids=input_ids, attention_mask=input_mask, segment_ids=segment_ids)
        loss = self.loss_func(logits, label_ids)
        return loss, logits, label_ids
        
    def training_step(self, batch, batch_idx):
        loss, logits, label_ids = self.calculate_loss(batch, batch_idx)
        self.log('train/loss', loss)
        return {'loss': loss, 'logits': logits, 'label_ids': label_ids}
    
    def training_step_end(self, step_output):
        logits = step_output['logits']
        label_ids = step_output['label_ids']
        self.train_label.extend(label_ids.cpu().detach().numpy().tolist())
        self.train_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    
    def training_epoch_end(self, outputs):
        train_accuracy = metrics.accuracy_score(self.train_label, self.train_predict)
        self.log('train/acc', train_accuracy, prog_bar=True)
        print("The training accuracy is ", train_accuracy)

    def validation_step(self, batch, batch_idx):
        val_loss, logits, label_ids = self.calculate_loss(batch, batch_idx)
        #self.log('val/loss', val_loss)
        return {'val_loss': val_loss, 'logits': logits, 'label_ids': label_ids}
    
    def validation_step_end(self, step_output):
        logits = step_output['logits']
        label_ids = step_output['label_ids']
        self.eval_targets.extend(label_ids.cpu().detach().numpy().tolist())
        self.eval_predict.extend(torch.max(logits, dim=1)[1].cpu().detach().numpy().tolist())
    
    def validation_epoch_end(self, outputs):
        mean_loss =torch.stack([x['val_loss'] for x in outputs]).mean() 
        eval_accuracy = metrics.accuracy_score(self.eval_targets, self.eval_predict)
        self.log('val/loss', mean_loss, prog_bar=True)
        self.log('val/acc', eval_accuracy, prog_bar=True)
        print('The validation accuracy is ', eval_accuracy)
        
        
    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {"params": self.roberta.parameters()},
            # {'params': model.highway.parameters(), 'lr': args.learning_rate * 10},
            {"params": self.classifier.parameters(), 'lr': self.args.learning_rate * 10}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        train_steps = len(self.train_dataloader()) // self.args.gradient_accumulation_steps \
                                                  * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0.05 * self.total_steps,
                                                    num_training_steps=train_steps)
        return [optimizer], [scheduler]
    
    