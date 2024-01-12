import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm

def inference(model, test_loader, sample_submission, now_time, args):
    model.eval()
    predicted_label_list = []

    test_loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for imgs, label in test_loop:
            imgs, label = imgs.float().to(args.device), label.float().to(args.device)
            # Forward & Loss
            predicted_label_list += nn.Sigmoid()(model(imgs)).tolist()
            
    if torch.cuda.device_count() > 1:
        print('multi-gpu')
        torch.save(model.module.state_dict(), f'{args.model_name}-{now_time}.pt')
    else:
        torch.save(model.state_dict(), f'{args.model_name}-{now_time}.pt')
    
    predicted = pd.DataFrame(predicted_label_list, columns=list(sample_submission.columns)[1:])
    result = pd.concat([sample_submission.img_id, predicted], axis=1)

    result.to_csv(f'{args.model_name}-{now_time}.csv', index=False)
    print(f"Inference completed and results saved to csv file.\nNow time: {now_time}")