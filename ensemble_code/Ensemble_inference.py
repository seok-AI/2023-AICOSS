import torch
from sklearn.metrics import average_precision_score
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def val_ensemble(models, val_loader, device, args):
    
    for model in models:
        model.eval()
    
    val_loop = tqdm(val_loader, leave=True)
    pred_np = []
    truth_np = []
    with torch.no_grad():
        for imgs, label in val_loop:
            if args.fast:
                imgs, label = imgs.float().half().to(device), label.float().half().to(device)
            else:
                imgs, label = imgs.float().to(device), label.float().to(device)
            # Forward & Loss
            pred_np_list = []
            for model in models:
                predicted_label = model(imgs)
                pred_np_model = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()
                pred_np_list.append(pred_np_model)
            
            
            
            label_np = label.cpu().detach().numpy()
            
            pred_np += (sum(pred_np_list) / len(models)).tolist()
            truth_np += label_np.tolist()
            
        total_val_map = average_precision_score(np.array(truth_np), np.array(pred_np))
         
    print(f"val map = {total_val_map:.4f}\n")
    return total_val_map

                
            
def test_ensemble(models, test_loader, device, args):    

    for model in models:
        model.eval()

    predicted_label_list = []

    test_loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for imgs, label in test_loop:
            if args.fast:
                imgs, label = imgs.float().half().to(device), label.float().half().to(device)
            else:
                imgs, label = imgs.float().to(device), label.float().to(device)
            
            # Forward & Loss
            pred_list = []
            for model in models:
                pred = nn.Sigmoid()(model(imgs))
                pred_list.append(pred)
            
            predicted_label_list += (sum(pred_list) / len(models)).tolist()
        
    return predicted_label_list



def val_ensemble_384(models, val_loader, device, args):
    
    for model in models:
        model.eval()
    
    val_loop = tqdm(val_loader, leave=True)
    pred_np = []
    truth_np = []
    with torch.no_grad():
        for imgs, label, img384 in val_loop:
            if args.fast:
                imgs, label, img384 = imgs.float().half().to(device), label.float().half().to(device), img384.float().half().to(device)
            else:
                imgs, label, img384 = imgs.float().to(device), label.float().to(device), img384.float().to(device)
                
            # Forward & Loss
            pred_np_list = []
            for model in models[:-1]:
                pred = nn.Sigmoid()(model(imgs)).squeeze(1).cpu().detach().numpy()
                pred_np_list.append(pred)
            
            pred = nn.Sigmoid()(models[-1](img384)).squeeze(1).cpu().detach().numpy()
            pred_np_list.append(pred)
                
            label_np = label.cpu().detach().numpy()
            
            pred_np += (sum(pred_np_list) / len(models)).tolist()
            truth_np += label_np.tolist()
            
        total_val_map = average_precision_score(np.array(truth_np), np.array(pred_np))
         
    print(f"val map = {total_val_map:.4f}\n")
    return total_val_map





def test_ensemble_384(models, test_loader, device, args):    

    for model in models:
        model.eval()

    predicted_label_list = []

    test_loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for imgs, label, img384 in test_loop:
            if args.fast:
                imgs, label, img384 = imgs.float().half().to(device), label.float().half().to(device), img384.float().half().to(device)
            else:
                imgs, label, img384 = imgs.float().to(device), label.float().to(device), img384.float().to(device)
            
            # Forward & Loss
            pred_list = []
            for model in models[:-1]:
                pred = nn.Sigmoid()(model(imgs))
                pred_list.append(pred)
            
            pred = nn.Sigmoid()(models[-1](img384))
            pred_list.append(pred)
            
            predicted_label_list += (sum(pred_list) / len(models)).tolist()
        
    return predicted_label_list
