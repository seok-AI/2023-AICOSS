import torch
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils_ import *

## Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(self.dataframe.iloc[idx][list(self.dataframe.columns)[2:]], dtype=torch.int8) if 'airplane' in self.dataframe.columns else 0
        
        return img, label


def My_DataLoader(train_data, args, val_data=None, test_data=None, num_workers=1):
    if args.augment == 'strong':
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)), 
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            CutoutPIL(cutout_factor=args.cutout_factor),
            transforms.AutoAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)), 
            CutoutPIL(cutout_factor=args.cutout_factor),
            transforms.RandAugment(args.magnitude),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CustomDataset(train_data, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = CustomDataset(val_data, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    test_dataset = CustomDataset(test_data, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
