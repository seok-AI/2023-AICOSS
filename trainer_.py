import torch
import wandb
from sklearn.metrics import average_precision_score
import numpy as np
from utils_ import *
from losses_ import *
from tqdm import tqdm

## Train
def val(model, criterion, val_loader, args):
    
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, leave=True)
    predict_np = []
    truth_np = []
    with torch.no_grad():
        for imgs, label in val_loop:
            imgs, label = imgs.float().to(args.device), label.float().to(args.device)
            # Forward & Loss
            predicted_label = model(imgs)
            loss = criterion(predicted_label.squeeze(1), label)

            label_np = label.cpu().detach().numpy()
            pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

            predict_np += pred_np.tolist()
            truth_np += label_np.tolist()
            
            val_loss += loss.item()
            val_loop.set_description(f"Validation")
            val_loop.set_postfix(loss=loss.item())
            
        total_val_map = average_precision_score(np.array(truth_np), np.array(predict_np))
        
    total_val_loss = val_loss / len(val_loader)
    print(f"val loss: {total_val_loss:.4f} val map = {total_val_map:.4f}\n")
    
    return total_val_loss, total_val_map



def train(model, criterion, optimizer, train_loader, args, scheduler=None, val_loader=None, Wandb=True):
    if Wandb:
        wandb.init(project=args.project, name= args.model_name + '-' +args.loss_name, notes=str(torch.cuda.get_device_name())+' x '+str(1))
    # 학습

    for epoch in range(args.epochs):
        predict_np, truth_np = [], []
        model.train()
        total_loss, total_train_map = 0, 0
        loop = tqdm(train_loader, leave=True)
        for imgs, label in loop:
            imgs, label = imgs.float().to(args.device), label.float().to(args.device)
            
            # Forward & Loss
            predicted_label = model(imgs)

            loss = criterion(predicted_label.squeeze(1), label)
            
            label_np = label.cpu().detach().numpy()
            pred_np = nn.Sigmoid()(predicted_label.squeeze(1)).cpu().detach().numpy()

            predict_np += pred_np.tolist()
            truth_np += label_np.tolist()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_description(f"Train")
            loop.set_postfix(loss=loss.item())
            
            
        total_train_map = average_precision_score(np.array(truth_np), np.array(predict_np))
        
        print(f"Epoch {epoch + 1} train loss: {total_loss / len(train_loader):.4f} train map = {total_train_map:.4f}")


        if scheduler != None:
            scheduler.step()
            
        if val_loader != None:
            val_loss, val_map = val(model, criterion, val_loader, args)
        else:
            val_loss, val_map = 0, 0
        
        if Wandb:
            wandb.log({"Epoch": epoch + 1,
                   "lr": optimizer.param_groups[0]["lr"],
                   "train loss": total_loss / len(train_loader),
                   "train map": total_train_map,
                   "val loss": val_loss,
                   "val map": val_map})   
