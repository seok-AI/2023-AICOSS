import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import warnings
from datetime import datetime
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import random

import torch.nn.functional as F
import timm


mlist = [0,0,0,['cvt_q2l', 'tresnet_xl_learnable_mldecoder', 'tresnetv2_l_mldecoder'],0,0,
         ['cvt_q2l', 'tresnet_xl_learnable_mldecoder', 'tresnetv2_l_mldecoder','tresnet_xl_learnable_mldecoder','tresnet_xl_mldecoder','tresnet_xl_mldecoder'],
         ['cvt_q2l', 'tresnet_xl_learnable_mldecoder', 'tresnetv2_l_mldecoder','tresnet_xl_learnable_mldecoder','tresnet_xl_mldecoder','tresnet_xl_mldecoder', 
          'cvt384_q2l']]

wlist = [0,0,0,['cvt_q2l-pasl.pt', 'tresnet_xl_learnable_mldecoder-min_lr-1e-5.pt', 'tresnetv2_l_mldecoder-zlpr.pt'],0,0,
         ['cvt_q2l-pasl.pt', 'tresnet_xl_learnable_mldecoder-min_lr-1e-5.pt', 'tresnetv2_l_mldecoder-zlpr.pt',
          'tresnet_xl_learnable_mldecoder-AutoAugment-H-V.pt','tresnet_xl_mldecoder-pasl.pt','tresnet_xl_mldecoder-two-loss.pt'],
         ['cvt_q2l-pasl.pt', 'tresnet_xl_learnable_mldecoder-min_lr-1e-5.pt', 'tresnetv2_l_mldecoder-zlpr.pt',
          'tresnet_xl_learnable_mldecoder-AutoAugment-H-V.pt','tresnet_xl_mldecoder-pasl.pt','tresnet_xl_mldecoder-two-loss.pt',
          'cvt384_q2l-GradAccum8.pt']]

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--seed', default=41, type=int)
parser.add_argument('--gpu', default='0', type=str)

parser.add_argument('--num_ensemble', default=3, type=int, choices=[3,6,7])
parser.add_argument('--path', default='/data/')
parser.add_argument('--weight_path', default='/2023-AICOSS/ensemble_code/weights/')
parser.add_argument('--fast', action="store_true")

args = parser.parse_args()


warnings.filterwarnings(action='ignore') 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())


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



from models_ import *
from Ensemble_inference import *



models = []

for i in range(args.num_ensemble):
    models.append(globals()[mlist[args.num_ensemble][i]]())
    
for i in range(args.num_ensemble):
    models[i].load_state_dict(torch.load(args.weight_path + wlist[args.num_ensemble][i]))
    
    if args.fast:
        models[i] = models[i].half().to(device)
    else:
        models[i] = models[i].to(device)
        
if args.fast:
    torch.set_float32_matmul_precision("medium")
    
    

## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None, transform2=None):
        self.dataframe = dataframe
        self.transform = transform
        self.transform2 = transform2
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
            
        # mos column 존재 여부에 따라 값을 설정
        label = torch.tensor(self.dataframe.iloc[idx][list(self.dataframe.columns)[2:]], dtype=torch.int8) if 'airplane' in self.dataframe.columns else 0
        
        if self.transform2:
            img224 = self.transform(img)
            img384 = self.transform2(img)
            return img224, label, img384
        else:
            img224 = self.transform(img)
            return img224, label


def main():
    
    # 데이터 로드
    path = args.path
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    sample_submission = pd.read_csv(path + 'sample_submission.csv')

    def rewrite(df):
        df['img_path'] = path + df['img_path']
        return df

    train = rewrite(train)
    test_data = rewrite(test)
    _, val_data = train_test_split(train, test_size=0.1, random_state=78, shuffle=True)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_384 = transforms.Compose([
        transforms.Resize((384, 384)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Ensemble
    
    if args.num_ensemble == 7:
        val_dataset_384 = CustomDataset(val_data, transform, transform_384)
        val_loader_384 = DataLoader(val_dataset_384, batch_size=64, shuffle=False, num_workers=4)

        test_dataset_384 = CustomDataset(test_data, transform, transform_384)
        test_loader_384 = DataLoader(test_dataset_384, batch_size=64, shuffle=False, num_workers=4)
        val_score = val_ensemble_384(models, val_loader_384, device, args=args)
        predicted_label_list = test_ensemble_384(models, test_loader_384, device, args=args)
        
        
    else:
        val_dataset = CustomDataset(val_data, transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        test_dataset = CustomDataset(test_data, transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        val_score = val_ensemble(models, val_loader, device, args=args)
        predicted_label_list = test_ensemble(models, test_loader, device, args=args)
    
    now = datetime.now()
    now_time = now.strftime("%m%d_%H%M")
    
    predicted = pd.DataFrame(predicted_label_list, columns=list(sample_submission.columns)[1:])
    result = pd.concat([sample_submission['img_id'], predicted], axis=1)

    result.to_csv(f'Ensemble{args.num_ensemble}-{val_score:.5f}-{now_time}.csv', index=False)
    print(f"Inference completed and results saved to csv file.\nnow time: {now_time}")

    
if __name__ == '__main__':
    main()