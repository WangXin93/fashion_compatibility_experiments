import logging
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from polyvore_dataset_name import TripletDataset, CategoryDataset
from csn import ConditionalSimNet
from tripletnet import CS_Tripletnet
from Resnet_18 import resnet18
from evaluate_csn import test_compatibility_auc, test_fitb_quesitons

img_size = 112
emb_size = 64
device = torch.device("cuda")

"""
A Train Dataset load anchor, positive, negative images, and condition
A Test Dataset load pair images, target and condition, conditions are:

    Conditions
    upper_bottom
    upper_shoe
    upper_bag
    upper_accessory
    bottom_shoe
    bottom_bag
    bottom_accessory
    shoe_bag
    shoe_accessory
    bag_accessory
"""
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Scale((img_size, img_size)),
        torchvision.transforms.CenterCrop(112),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)
train_dataset = TripletDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    data_file="train_no_dup_with_category_3more_name.json",
)
train_loader = DataLoader(train_dataset, 32, shuffle=True, num_workers=4)
val_dataset = TripletDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    data_file="valid_no_dup_with_category_3more_name.json",
    is_train=True,
)
val_loader = DataLoader(val_dataset, 32, shuffle=False, num_workers=4)
test_dataset = TripletDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    data_file="test_no_dup_with_category_3more_name.json",
    is_train=True,
)
test_loader = DataLoader(test_dataset, 32, shuffle=False, num_workers=4)

val_auc_dataset = CategoryDataset(
    transform=transform,
    use_mean_img=True,
    data_file="valid_no_dup_with_category_3more_name.json",
    neg_samples=True,
)

# An encoding Net, CSN_Net, Triplet Net
model = resnet18(pretrained=True, embedding_size=emb_size)
csn_model = ConditionalSimNet(
    model,
    n_conditions=len(train_dataset.conditions) // 2,
    embedding_size=emb_size,
    learnedmask=True,
    prein=False,
)
tnet = CS_Tripletnet(csn_model)
tnet = tnet.to(device)

# Hyperparameters
criterion = torch.nn.MarginRankingLoss(margin=0.2)
parameters = filter(lambda p: p.requires_grad, tnet.parameters())
optimizer = torch.optim.Adam(parameters, lr=5e-5)

n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
print(f" + Number of params: {n_parameters}")


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum().item() / dista.size(0)


best_acc = -1
for epoch in range(1, 50 + 1):
    print(f"** Epoch: {epoch} **")
    # Train
    tnet.train()
    losses = AverageMeter()
    accs = AverageMeter()
    for batch_num, (a_img, p_img, n_img, c) in enumerate(train_loader, 1):
        a_img, p_img, n_img, c = (
            a_img.to(device),
            p_img.to(device),
            n_img.to(device),
            c.to(device),
        )
        # Original code need input triplet like: anchor, far, close
        dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(
            a_img, n_img, p_img, c
        )

        target = torch.FloatTensor(dista.size()).fill_(1).to(device)

        loss_triplet = criterion(dista, distb, target)
        loss_embed = embed_norm / np.sqrt(a_img.size(0))
        loss_mask = mask_norm / a_img.size(0)
        loss = loss_triplet + 5e-3 * loss_embed + 5e-4 * loss_mask

        losses.update(loss_triplet.item(), a_img.size(0))
        accs.update(accuracy(dista, distb), a_img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_num % 50 == 0:
            print(
                f"#{batch_num} Loss: {losses.val:.4f} (avg: {losses.avg:.4f}), Accuracy: {accs.avg:.4f}"
            )

    print(f"Train Loss: {losses.avg:.4f}, Accuracy: {accs.avg:.4f}")

    # Evaluation
    tnet.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    for batch_num, (a_img, p_img, n_img, c) in enumerate(val_loader, 1):
        a_img, p_img, n_img, c = (
            a_img.to(device),
            p_img.to(device),
            n_img.to(device),
            c.to(device),
        )
        # Original code need input triplet like: anchor, far, close
        with torch.no_grad():
            dista, distb, mask_norm, embed_norm, mask_embed_norm = tnet(
                a_img, n_img, p_img, c
            )

        target = torch.FloatTensor(dista.size()).fill_(1).to(device)

        loss_triplet = criterion(dista, distb, target)

        losses.update(loss_triplet.item(), a_img.size(0))
        accs.update(accuracy(dista, distb), a_img.size(0))

    # Valid AUC
    auc = test_compatibility_auc(val_auc_dataset, tnet.embeddingnet)
    acc = test_fitb_quesitons(val_auc_dataset, tnet.embeddingnet)

    print(
        f"Valid Loss: {losses.avg:.4f}, Accuracy: {accs.avg:.4f}, AUC: {auc:.4f}, FitbACC: {acc:.4f}"
    )

    # Save Model
    if accs.avg > best_acc:
        best_acc = accs.avg
        torch.save(tnet.state_dict(), "csn_model_best.pth")
        print(f"Found Best Accuracy {accs.avg:.4f}, saved model to csn_model_best.pth")
