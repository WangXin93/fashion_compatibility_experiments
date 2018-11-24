import torchvision
import logging
import os
import pickle
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from model import EncoderCNN, LSTMModel
from polyvore_dataset_name import CategoryDataset, lstm_collate_fn
from torch.utils.data import DataLoader

batch_size = 32
model = "lstm"
img_size=299
emb_size = 512
device = torch.device("cuda")

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)
test_dataset = CategoryDataset(
    transform=transform,
    data_file="test_no_dup_with_category_3more_name.json",
    use_mean_img=False,
    neg_samples=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=lstm_collate_fn,
)

encoder_cnn = EncoderCNN(emb_size)
encoder_cnn.load_state_dict(torch.load("./encoder_cnn.pth"))
print("Successfully load trained weights...")
encoder_cnn = encoder_cnn.to(device)
encoder_cnn.train(False)

test_features = {}
for batch_num, input_data in enumerate(test_loader, 1):
    print("#{}\r".format(batch_num), end="")
    lengths, images, names, offsets, set_ids, labels, is_compat = input_data

    image_seqs = images.to(device)
    with torch.no_grad():
        emb_seqs = encoder_cnn(image_seqs)

    batch_ids = []
    for set_id, items in zip(set_ids, labels):
        for item in items:
            batch_ids.append(item)

    for i, id in enumerate(batch_ids):
        test_features[id] = emb_seqs[i].cpu().detach().numpy()
print()

pickle.dump(test_features, open("test_features.pkl", "wb"))
print("Done.")
