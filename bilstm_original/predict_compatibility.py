from polyvore_dataset import PredictCompatibilityDataset
from model import GRUModel, EncoderCNN, LSTMModel
from polyvore_dataset import create_dataloader
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import logging
import torch
import os
import sys
import tqdm
import pickle
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

device = torch.device("cuda")

def flip_tensor(tensor, device=device):
    """Flip a tensor in 0 dim for backward rnn.
    """
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    flipped_tensor = tensor.index_select(0, idx)
    return flipped_tensor

# Load pretrained feature
test_features = pickle.load(open('./test_features.pkl', 'rb'))
test_features_ids = []
test_features_matrix = []
for k, v in test_features.items():
	test_features_matrix.append(v)
	test_features_ids.append(k)
test_features_matrix = torch.tensor(np.stack(test_features_matrix)).to(device)

emb_size = 512

dataset = PredictCompatibilityDataset()

# Restore model parameters
encoder_cnn = EncoderCNN(emb_size)
# encoder_cnn.load_state_dict(torch.load('./runs/lstm-xv3-bn-bi-alllayer/encoder_cnn-30.ckpt'))
encoder_cnn.load_state_dict(torch.load('./encoder_cnn_vse.pth'))
encoder_cnn = encoder_cnn.to(device)

f_rnn = LSTMModel(emb_size, emb_size, emb_size, device,  bidirectional=False)
b_rnn = LSTMModel(emb_size, emb_size, emb_size, device,  bidirectional=False)
# f_rnn.load_state_dict(torch.load('./runs/lstm-xv3-bn-bi-alllayer/f_rnn-30.ckpt'))
# b_rnn.load_state_dict(torch.load('./runs/lstm-xv3-bn-bi-alllayer/b_rnn-30.ckpt'))
f_rnn.load_state_dict(torch.load('./f_rnn_vse.pth'))
b_rnn.load_state_dict(torch.load('./b_rnn_vse.pth'))
f_rnn = f_rnn.to(device)
b_rnn = b_rnn.to(device)

criterion = nn.CrossEntropyLoss()

encoder_cnn.train(False)
f_rnn.train(False)
b_rnn.train(False)
f_losses, b_losses, truths = [], [], []

for idx in tqdm.trange(len(dataset)):
    input_data = dataset[idx]
    images, image_ids, label = input_data
    lengths = torch.tensor([len(image_ids)-1]).to(device)
    image_seqs = images.to(device)
    
    with torch.no_grad():
        emb_seqs = encoder_cnn(image_seqs)
    f_emb_seqs = emb_seqs[:-1] # (1, 2, 3, 4)
    b_emb_seqs = flip_tensor(emb_seqs[1:]) # (5, 4, 3, 2)
    f_emb_seqs = f_emb_seqs.unsqueeze(dim=0)
    b_emb_seqs = b_emb_seqs.unsqueeze(dim=0)

    with torch.no_grad():
        f_output = f_rnn(f_emb_seqs, lengths)
        b_output = b_rnn(b_emb_seqs, lengths)

    f_score = torch.matmul(f_output, test_features_matrix.t())
    b_score = torch.matmul(b_output, test_features_matrix.t())
    f_targets = [test_features_ids.index(i) for i in image_ids[1:]] # (2, 3, 4, 5) 
    b_targets = [test_features_ids.index(i) for i in image_ids[:-1][::-1]] # (4, 3, 2, 1)
    f_loss = criterion(f_score, torch.tensor(f_targets).to(device))
    b_loss = criterion(b_score, torch.tensor(b_targets).to(device))

    f_losses.append(f_loss.item())
    b_losses.append(b_loss.item())
    truths.append(int(label))

f_losses, b_losses, truths = np.array(f_losses), np.array(b_losses), np.array(truths)
f_auc = metrics.roc_auc_score(truths, -f_losses)
b_auc = metrics.roc_auc_score(truths, -b_losses)
all_auc = metrics.roc_auc_score(truths, -f_losses-b_losses)
print('F_AUC: {:.4f}, B_AUC: {:.4f}, ALL_AUC: {:.4f}'.format(f_auc, b_auc, all_auc))

# lstm-xv3-bn-bi-alllayer, epoch 30: F_AUC: 0.9001, B_AUC: 0.9042, ALL_AUC: 0.9065
# lstm-xv3-dropout-bi-alllayer, epoch 30: F_AUC: 0.85, B_AUC: 0.85, ALL_AUC: 0.85
