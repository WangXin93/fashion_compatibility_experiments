import logging
import numpy as np
import resnet
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision
from model import SigmoidC
from polyvore_dataset_name import CategoryDataset, collate_fn
from sklearn import metrics
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Leave a comment for this training, and it will be used for name suffix of log and saved model
if sys.argv[1:]:
    comment = sys.argv[1]
else:
    comment = ""

# Logger
log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"
logging.basicConfig(
    level=logging.INFO,
    filename="log_{}{}.log".format(__file__.split(".")[0], comment),
    format=log_format,
)

# Dataloader
img_size = 224
device = torch.device("cuda:0")
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)
train_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    use_mean_img=True,
    data_file="train_no_dup_with_category_3more_name.json",
)
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn
)
val_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    use_mean_img=True,
    data_file="valid_no_dup_with_category_3more_name.json",
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn
)
test_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform,
    use_mean_img=True,
    data_file="test_no_dup_with_category_3more_name.json",
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn
)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Model!!!!!!!!!!!
class CompatModel(nn.Module):
    def __init__(self, embed_size=1000, need_rep=False):
        """Load the pretrained CNN and replace top fc layer.
        Args:
            embed_size: the output embedding size of the cnn model, default 1000.
            need_rep: whether need representation of the layer before last fc 
                layer, whose size is 2048.
        """
        super(CompatModel, self).__init__()
        cnn = resnet.resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.need_rep = need_rep
        self.cnn = cnn
        self.bn = nn.BatchNorm1d(25)  # 5x5 relationship matrix have 25 elements
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = SigmoidC()

        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

        # Type specified masks
        self.masks = nn.Embedding(embed_size, embed_size)
        self.masks.weight.data.normal_(0.9, 0.7)

    def forward(self, images):
        """Extract feature vectors from input images."""
        batch_size = images.shape[0]
        item_num = 5
        images = torch.reshape(images, (-1, 3, img_size, img_size))
        if self.need_rep:
            features, rep = self.cnn(images)
        else:
            features = self.cnn(images)
        features = features.reshape(batch_size, item_num, -1)  # (32, 5, 1000)

        # Type specified representation
        masked = []
        masks = []
        for i in range(item_num):
            mask = F.relu(self.masks(torch.tensor(i).to(device)))
            masks.append(mask)
            masked.append(mask * features[:, i, :])
        masked = torch.stack(masked, dim=1)
        masks = torch.stack(masks, dim=0)

        # Non-local like matmul to construct relationship matrix
        cross = torch.matmul(masked, masked.transpose(1, 2))  # (32, 5, 5)
        cross = self.bn(cross.view(batch_size, -1))

        out = F.relu(self.fc1(cross), inplace=True)
        out = self.sigmoid(self.fc2(out))
        if self.need_rep:
            return out, features, masks, rep
        else:
            return out, features, masks


# Semantic embedding model
embedding = nn.Embedding(len(train_dataset.vocabulary), 1000)
embedding = embedding.to(device)

# Visual embedding model
image_embedding = nn.Linear(2048, 1000)
image_embedding = image_embedding.to(device)

# Train!!!!!!!!!!
model = CompatModel(embed_size=1000, need_rep=True)
model = model.to(device)
criterion = nn.BCELoss()
parameters = (
    list(model.parameters())
    + list(embedding.parameters())
    + list(image_embedding.parameters())
)
optimizer = torch.optim.SGD(parameters, lr=1e-2, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
best_auc = -1
for epoch in range(1, 50 + 1):
    logging.info("Train Phase, Epoch: {}".format(epoch))
    scheduler.step()
    total_losses = AverageMeter()
    clf_losses = AverageMeter()
    vse_losses = AverageMeter()

    # Train phase
    model.train()
    for batch_num, batch in enumerate(train_loader, 1):
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        # Normalized Semantic Embedding
        padded_names = rnn_utils.pad_sequence(names, batch_first=True).to(device)
        mask = torch.gt(padded_names, 0)
        cap_mask = torch.ge(mask.sum(dim=1), 2)
        semb = embedding(padded_names)
        semb = semb * (mask.unsqueeze(dim=2)).float()
        word_lengths = mask.sum(dim=1)
        word_lengths = torch.where(
            word_lengths == 0,
            (torch.ones(semb.shape[0]).float() * 0.1).to(device),
            word_lengths.float(),
        )
        semb = semb.sum(dim=1) / word_lengths.unsqueeze(dim=1)
        semb = F.normalize(semb, dim=1)

        # Normalized Visual Embedding
        images = images.to(device)
        output, features, tmasks, rep = model(images)
        vemb = F.normalize(image_embedding(rep), dim=1)

        # Type embedding loss
        tmasks_loss = tmasks.norm(1) / len(tmasks)
        features_loss = features.norm(2) / np.sqrt(
            (features.shape[0] * features.shape[1])
        )

        # BCE Loss
        target = is_compat.float().to(device)
        output = output.squeeze(dim=1)
        clf_loss = criterion(output, target)

        # VSE Loss
        # Reference: https://github.com/xthan/polyvore/blob/e0ca93b0671491564b4316982d4bfe7da17b6238/polyvore/polyvore_model_bi.py#L362
        semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
        vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
        semb = semb.reshape([-1, 1000])
        vemb = vemb.reshape([-1, 1000])
        scores = torch.matmul(semb, vemb.transpose(0, 1))
        diagnoal = scores.diag().unsqueeze(dim=1)
        cost_s = torch.clamp(0.2 - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
        cost_im = torch.clamp(0.2 - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
        cost_s = cost_s - torch.diag(cost_s.diag())
        cost_im = cost_im - torch.diag(cost_im.diag())
        vse_loss = cost_s.sum() + cost_im.sum()
        vse_loss = vse_loss / (semb.shape[0] ** 2)

        # Sum all losses up
        features_loss = 5e-3 * features_loss
        tmasks_loss = 5e-4 * tmasks_loss
        total_loss = clf_loss + vse_loss + features_loss + tmasks_loss

        # Update Recoder
        total_losses.update(total_loss.item(), images.shape[0])
        clf_losses.update(clf_loss.item(), images.shape[0])
        vse_losses.update(vse_loss.item(), images.shape[0])

        # Backpropagation
        model.zero_grad()
        total_loss.backward()
        optimizer.step()
        if batch_num % 2 == 0:
            logging.info(
                f"#{batch_num} clf_loss: {clf_losses.val:.4f}, vse_loss: {vse_losses.val:.4f}, features_loss: {features_loss:.4f}, tmasks_loss: {tmasks_loss:.4f}, total_loss:{total_losses.val:.4f}"
            )
    logging.info(f"Train Loss (clf_loss): {clf_losses.avg:.4f}")

    # Valid Phase
    logging.info("Valid Phase, Epoch: {}".format(epoch))
    model.eval()
    clf_losses = AverageMeter()
    outputs = []
    targets = []
    for batch_num, batch in enumerate(val_loader, 1):
        lengths, images, names, offsets, set_ids, labels, is_compat = batch
        images = images.to(device)
        target = is_compat.float().to(device)
        with torch.no_grad():
            output, _, _, _ = model(images)
            output = output.squeeze(dim=1)
            clf_loss = criterion(output, target)
        clf_losses.update(clf_loss.item(), images.shape[0])
        outputs.append(output)
        targets.append(target)
    logging.info(f"Valid Loss (clf_loss): {clf_losses.avg:.4f}")
    outputs = torch.cat(outputs).cpu().data.numpy()
    targets = torch.cat(targets).cpu().data.numpy()
    auc = metrics.roc_auc_score(targets, outputs)
    logging.info("AUC: {:.4f}".format(auc))
    predicts = np.where(outputs > 0.5, 1, 0)
    accuracy = metrics.accuracy_score(predicts, targets)
    logging.info("Accuracy@0.5: {:.4f}".format(accuracy))

    # Save best model
    save_path = "model" + comment + ".pth"
    save_emb_path = "emb_model" + comment + ".pth"
    save_imgemb_path = "imgemb_model" + comment + ".pth"
    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), save_path)
        torch.save(embedding.state_dict(), save_emb_path)
        torch.save(image_embedding.state_dict(), save_emb_path)
        logging.info("Saved model to {}".format(save_path))
        logging.info("Saved embedding model to {}".format(save_emb_path))
        logging.info("Saved image embedding model to {}\n".format(save_imgemb_path))
