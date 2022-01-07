import torch
from torch.nn.modules.loss import _Loss
from VGG_FCN.model import *
from utils import *
from dataset import *
from torch.utils.data import Dataset,DataLoader 
import os
import time
import cv2
import wandb

best_iou = 0

def main():
    global best_iou
    args = parse_args()
    start_epoch = 0
    save_path = args.save
    im_path = args.image
    json_path = args.json
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(im_path):
        os.makedirs(im_path)
    if not os.path.isdir(json_path):
        os.makedirs(json_path)
    train_dataset = LaneDataSet(args.dataset,"train")
    val_dataset = LaneDataSet(args.dataset,"val")
    model = LaneVggFCNAttNet().cuda()
    #load_imagenet(model)
    train_loader = DataLoader(train_dataset, batch_size= 8, pin_memory = True, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size= 8, pin_memory = True, shuffle = True)
    if args.pretrained:
        saved = torch.load(args.pretrained)
        model.load_state_dict({k:v for k,v in saved.items() if k in model.state_dict()})
        start_epoch = int(args.pretrained.split("_")[-2])
    
    torch.backends.cudnn.benchmark = True
    # optimizer = torch.optim.SGD(model.parameters(),lr = state['lr'],momentum = 0.9,weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
    wandb.init(project='Lane_Segmentation',name="VGG_FCN_baseline")
    wandb.watch(model)
    for epoch in range(start_epoch,args.epochs):
        #  adjust_learning_rate(optimizer,epoch)
        print(f'Starting Epoch....{epoch}')
        train_iou = train(train_loader,model,optimizer,im_path,epoch)
        val_iou = test(val_loader,model,im_path,json_path,epoch)
        wandb.log({
        "Epoch": epoch,
        "Train IOU": train_iou,
        "Valid IOU": val_iou,
        })
        if (epoch+1)%5 == 0:
            pass
            #save_model(save_path,epoch,model)
        best_iou = max(val_iou,best_iou)
        print('Best IoU : {}'.format(best_iou))

if __name__ == '__main__':
    main()
    
    
