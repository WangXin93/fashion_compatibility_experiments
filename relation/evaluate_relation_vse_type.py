import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from polyvore_dataset_name import CategoryDataset, collate_fn
from sklearn import metrics
import torch.nn.functional as F
import resnet
from model import SigmoidC

# Dataloader
img_size = 224
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)
train_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform, data_file="train_no_dup_with_category_3more_name.json"
)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn
)
val_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform, data_file="valid_no_dup_with_category_3more_name.json"
)
val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn
)
test_dataset = CategoryDataset(
    root_dir="/home/wangx/datasets/polyvore/images/",
    transform=transform, data_file="test_no_dup_with_category_3more_name.json"
)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn
)

# Model!!!!!!!!!!!
class CompatModel(nn.Module):
    def __init__(self, embed_size, need_rep=False):
        """Load the pretrained CNN and replace top fc layer."""
        super(CompatModel, self).__init__()
        cnn = resnet.resnet50(pretrained=True, need_rep=need_rep)
        cnn.fc = nn.Linear(cnn.fc.in_features, embed_size)
        self.need_rep = need_rep
        self.cnn = cnn
        self.bn = nn.BatchNorm1d(25)
        self.fc1 = nn.Linear(25, 25)
        self.fc2 = nn.Linear(25, 1)
        self.sigmoid = SigmoidC()

        nn.init.xavier_uniform_(cnn.fc.weight)
        nn.init.constant_(cnn.fc.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

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

        # Non-local like matmul
        cross = torch.matmul(masked, masked.transpose(1, 2))  # (32, 5, 5)
        cross = self.bn(cross.view(batch_size, -1))

        out = F.relu(self.fc1(cross), inplace=True)
        out = self.sigmoid(self.fc2(out))
        if self.need_rep:
            return out, features, masks, rep
        else:
            return out, features, masks


# Test!!!!!!!!!!
device = torch.device("cuda:0")
model = CompatModel(embed_size=1000, need_rep=True).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# Compatibility AUC test
model.load_state_dict(torch.load("./model.pth"))
model.eval()
total_loss = 0
outputs = []
targets = []
for batch_num, batch in enumerate(test_loader, 1):
    print("\r#{}".format(batch_num), end="", flush=True)
    lengths, images, names, offsets, set_ids, labels, is_compat = batch
    images = images.to(device)
    target = is_compat.float().to(device)
    with torch.no_grad():
        output, _, _, _ = model(images)
        output = output.squeeze(dim=1)
        loss = criterion(output, target)
    total_loss += loss.item()
    outputs.append(output)
    targets.append(target)
print()
print("Valid Loss: {:.4f}".format(total_loss / batch_num))
outputs = torch.cat(outputs).cpu().data.numpy()
targets = torch.cat(targets).cpu().data.numpy()
print("AUC: {:.4f}".format(metrics.roc_auc_score(targets, outputs)))


# Fill in the blank evaluation
is_correct = []
for i in range(len(test_dataset)):
    print("\r#{}".format(i), end="", flush=True)
    items, labels, question_part, question_id, options, option_labels = test_dataset.get_fitb_quesiton(
        i
    )
    question_part = {"upper": 0, "bottom": 1, "shoe": 2, "bag": 3, "accessory": 4}.get(
        question_part
    )
    images = [items]

    for option in options:
        new_outfit = items.clone()
        new_outfit[question_part] = option
        images.append(new_outfit)
    images = torch.stack(images).to(device)
    output, _, _, _ = model(images)

    if output.argmax().item() == 0:
        is_correct.append(True)
    else:
        is_correct.append(False)
print()
print("FitB ACC: {:.4f}".format(sum(is_correct) / len(is_correct)))
