import argparse

def set_args(Trainer):
    parser = argparse.ArgumentParser('classification')
    parser.add_argument('--pretrained_model_path', default='./pretrain', type=str, help='预训练模型的路径')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='模型输出')
    parser.add_argument('--num_train_epochs', default=1, type=int, help='训练几轮')
    parser.add_argument('--train_batch_size', default=16, type=int, help='训练批次大小')
    parser.add_argument('--val_batch_size', default=16, type=int, help='验证批次大小')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='梯度积累几次更新')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='学习率大小')
    parser.add_argument('--seed', default=43, type=int, help='随机种子')
    parser.add_argument('--label2id_path', default='./data/label2id.json')
    parser.add_argument('--train_data_path', default='./data/train.json')
    parser.add_argument('--dev_data_path', default='./data/dev.json')
    parser.add_argument('--num_gpus', type=int, default=0)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    return args