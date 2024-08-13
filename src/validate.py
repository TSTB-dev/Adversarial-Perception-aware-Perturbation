import torch
from torch import nn
import wandb

from utils import LogMeter

def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    args,
):
    model.eval()
    loss_meter = LogMeter()
    acc_meter = LogMeter()
    loss_meter.reset()
    acc_meter.reset()
    
    for i, batch in enumerate(dataloader):
        imgs, labels = batch["image"], batch["label"]
        imgs = imgs.cuda()
        labels = labels.cuda()
        with torch.inference_mode():
            preds = model(imgs)
            loss = nn.CrossEntropyLoss()(preds, labels)
        
        loss_meter.update(loss.item())
        acc_meter.update((preds.argmax(1) == labels).float().mean().item())
        
    wandb.log(
        {"val/loss": loss_meter.avg, 
            "val/acc": acc_meter.avg
        }
    )
    print(f"(Validation) [Epoch {epoch}] loss: {loss_meter.avg} acc: {acc_meter.avg}")
    
    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg
    }

        