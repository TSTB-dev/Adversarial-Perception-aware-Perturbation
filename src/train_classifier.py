import torch
from torch import nn
import wandb

from utils import LogMeter


import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision

from dataset import PetsDataset
from models import get_model
from validate import validate

import logging
import wandb
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="pets")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_channel", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--optimzier", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--scheduler_epochs", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="apap")
    parser.add_argument("--wandb_run_name", type=str, default="apap-pets-real-resnet50")
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    args = parser.parse_args()
    config = vars(args)
    logging.info(f"Config: {config}")
    
    # dataset 
    dataset_config = {
        "root": args.data_path,
        "download": True,
        "transform": None,
        "train": True,
    }
    if args.dataset == "pets":
        pets_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.image_size, args.image_size)),
                torchvision.transforms.ToTensor(),
            ]
        )
        dataset_config["transform"] = pets_transform
        train_dataset = PetsDataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = PetsDataset(**dataset_config)        
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # model
    model = get_model(args, args.model_name, in_channels=args.image_channel, num_classes=train_dataset.num_classes)

    model = model.cuda()
    logging.info(f"Model: {model}")
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # scheduler
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.scheduler_epochs)
    else:
        raise ValueError(f"Invalid scheduler: {args.scheduler}")
    
    # logging
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
    
    logging.info(f"Start training for {args.epochs} epochs")
    for i in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_dataloader, optimizer, scheduler, i, args)
        
        if i % 10 == 0:
            val_stats = validate(model, val_dataloader, i, args)
            logging.info(f"[Epoch {i}] val_loss: {val_stats['loss']}, val_acc: {val_stats['acc']}")
    
    logging.info(f"Overall Training finished")
    
    if args.save_model:
        save_dir = os.path.join(wandb.run.dir, "models")
        Path(save_dir).mkdir(parents=True, exist_ok=True)   
        logging.info(f"Save model to {save_dir}")
        torch.save(model.state_dict(), os.path.join(save_dir, "classifier.pth"))
    
    wandb.finish()
        
    
def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    args
):
    model.train()
    loss_meter = LogMeter()
    acc_meter = LogMeter()
    acc_meter.reset()
    loss_meter.reset()
    
    for i, batch in enumerate(dataloader):
        imgs, labels = batch["image"], batch["label"]
        imgs = imgs.cuda()
        labels = labels.cuda()
        
        preds = model(imgs)
        
        loss = nn.CrossEntropyLoss()(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        acc_meter.update((preds.argmax(1) == labels).float().mean().item())
        loss_meter.update(loss.item())
        
        if i % 100 == 0:
            wandb.log(
                {"train/loss": loss.item()}
            )
            print(f"[Epoch {epoch}] Step {i}: loss: {loss.item()}")

    print(f"[Epoch {epoch}] loss: {loss_meter.avg}, acc: {acc_meter.avg}")
        
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg
    }
        
if __name__ == "__main__":
    main()