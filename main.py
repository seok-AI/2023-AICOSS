import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np
import random
import warnings
from sklearn.model_selection import train_test_split
from datetime import datetime
import argparse

model_list = ['resnet101', 'resnet50', 'resnet50_mldecoder', 'swinv2', 'swinv2_mldecoder', 'cvt_q2l',
              'tresnet_l_learnable_mldecoder', 'tresnet_xl_learnable_mldecoder', 'tresnet_xl_mldecoder', 
                'tresnet_xl_q2l', 'tresnetv2_l_mldecoder', 'tresnetv2_l_learnable_mldecoder']

loss_list = ['AsymmetricLoss', 'AsymmetricLossOptimized', 'ComputePrior', 'FocalLoss', 
             'PartialSelectiveLoss', 'TwoWayLoss', 'focal_binary_cross_entropy', 
             'multilabel_categorical_crossentropy', 'zlpr', 'zlpr_smooth']


parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--min_lr', default=1e-6, type=float)
parser.add_argument('--magnitude', default=9, type=int)
parser.add_argument('--seed', default=41, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--augment', default='weak', type=str, choices = ['strong', 'weak'])

parser.add_argument('--model_name', default='tresnet_l_learnable_mldecoder', choices = model_list)
parser.add_argument('--path', default='/home/6210seok/UOS_DACON/DATA/')
parser.add_argument('--loss_name', default='PartialSelectiveLoss', choices = loss_list)
parser.add_argument('--project', default='SOTA')
parser.add_argument('--use_prior', action="store_true") # 입력 시 True
parser.add_argument('--use_wandb', action="store_true") # 입력 시 True


args = parser.parse_args()

warnings.filterwarnings(action='ignore') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())


parser.add_argument('--device', default=device, type=str)
if args.loss_name == 'PartialSelectiveLoss':
    # For PartialSelectiveLoss
    parser.add_argument('--num-classes', default=60)
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--image-size', default=224, type=int,
                        metavar='N', help='input image size (default: 448)')
    parser.add_argument('--simulate_partial_type', type=str, default=None, help="options are fpc or rps")
    parser.add_argument('--simulate_partial_param', type=float, default=1000)
    parser.add_argument('--partial_loss_mode', type=str, default="negative")
    parser.add_argument('--clip', type=float, default=0)
    parser.add_argument('--gamma_pos', type=float, default=1)
    parser.add_argument('--gamma_neg', type=float, default=1)
    parser.add_argument('--gamma_unann', type=float, default=2)
    parser.add_argument('--alpha_pos', type=float, default=1)
    parser.add_argument('--alpha_neg', type=float, default=1)
    parser.add_argument('--alpha_unann', type=float, default=1)
    parser.add_argument('--likelihood_topk', type=int, default=5)
    parser.add_argument('--prior_path', type=str, default=None)
    parser.add_argument('--prior_threshold', type=float, default=0.05)
    # parser.add_argument('-b', '--batch-size', default=160, type=int,
    #                     metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--print-freq', '-p', default=128, type=int,
                        metavar='N', help='print frequency (default: 64)')
    parser.add_argument('--path_dest', type=str, default="./outputs")
    parser.add_argument('--debug_mode', type=str, default="hyperml")


args = parser.parse_args()
    
## Fixed Random-Seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(args.seed) # Seed 고정

from losses_ import *
from models_ import *
from utils_ import *
from data_loader_ import *
from trainer_ import *
from inference_ import *

def main():

    path = args.path
    train_df = pd.read_csv(path + 'train.csv')
    test_df = pd.read_csv(path + 'test.csv')
    sample_submission = pd.read_csv(path + 'sample_submission.csv')

    def rewrite(df):
        df['img_path'] = path + df['img_path']
        return df

    train_df = rewrite(train_df)
    test_df = rewrite(test_df)
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=78, shuffle=True)

    model = globals()[args.model_name]()
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    
    if args.loss_name == 'PartialSelectiveLoss':
        criterion = globals()[args.loss_name](args) 
    else:
        criterion = globals()[args.loss_name]()
         
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    train_loader, val_loader, test_loader = My_DataLoader(train_data, args, val_data, test_df, num_workers=4)
    
    train(model, criterion, optimizer, train_loader, args, scheduler, val_loader, Wandb=args.use_wandb)
    
    
    now = datetime.now()
    now_time = now.strftime("%m%d_%H%M")
    
    inference(model, test_loader, sample_submission, now_time, args)
    
    
if __name__ == '__main__':
    main()
