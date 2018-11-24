import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torchvision
import logging
import torch
import os
import sys
import tqdm
import pickle
import numpy as np
from sklearn import metrics
from polyvore_dataset import categoryDataset
from model import EncoderCNN, LSTMModel

device = torch.device("cuda")
img_size = 299
emb_size = 512

def flip_tensor(tensor, dim=0, device=device):
    """Flip a tensor in 0 dim for backward rnn."""
    idx = [i for i in range(tensor.size(dim)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    flipped_tensor = tensor.index_select(dim, idx)
    return flipped_tensor

# Load pretrained feature
test_features = pickle.load(open('./test_features.pkl', 'rb'))
test_features_ids = []
test_features_matrix = []
for k, v in test_features.items():
    test_features_matrix.append(v)
    test_features_ids.append(k)
test_features_matrix = torch.tensor(np.stack(test_features_matrix)).to(device)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)
test_dataset = categoryDataset(
    transform=transform,
    data_file="test_no_dup_with_category_3more.json",
    use_mean_img=False,
    neg_samples=True
)

# Restore model parameters
encoder_cnn = EncoderCNN(emb_size)
#encoder_cnn.load_state_dict(torch.load('./models/encoder_cnn.pth'))
encoder_cnn.load_state_dict(torch.load('/export/home/wangx/code/pytorch-tutorial/tutorials/03-advanced/my_polyvore/runs/bilstm/encoder_cnn-30.ckpt'))
encoder_cnn = encoder_cnn.to(device)

f_rnn = LSTMModel(emb_size, emb_size, emb_size, device,  bidirectional=False)
b_rnn = LSTMModel(emb_size, emb_size, emb_size, device,  bidirectional=False)
#f_rnn.load_state_dict(torch.load('./models/f_rnn.pth'))
f_rnn.load_state_dict(torch.load('/export/home/wangx/code/pytorch-tutorial/tutorials/03-advanced/my_polyvore/runs/bilstm/f_rnn-30.ckpt'))
#b_rnn.load_state_dict(torch.load('./models/b_rnn.pth'))
b_rnn.load_state_dict(torch.load('/export/home/wangx/code/pytorch-tutorial/tutorials/03-advanced/my_polyvore/runs/bilstm/b_rnn-30.ckpt'))
f_rnn = f_rnn.to(device)
b_rnn = b_rnn.to(device)

encoder_cnn.train(False)
f_rnn.train(False)
b_rnn.train(False)

criterion = nn.CrossEntropyLoss()
# Compatibility AUC test
f_losses, b_losses, truths = [], [], []
for idx in tqdm.trange(len(test_dataset)):
    input_data = test_dataset[idx]
    images, set_ids, image_ids, is_compat = input_data
    lengths = torch.tensor([len(image_ids)-1]).to(device) # 1, 2, 3 predicts 2, 3, 4, so length-1
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
    truths.append(is_compat)

f_losses, b_losses, truths = np.array(f_losses), np.array(b_losses), np.array(truths)
f_auc = metrics.roc_auc_score(truths, -f_losses)
b_auc = metrics.roc_auc_score(truths, -b_losses)
all_auc = metrics.roc_auc_score(truths, -f_losses-b_losses)
print('F_AUC: {:.4f}, B_AUC: {:.4f}, ALL_AUC: {:.4f}'.format(f_auc, b_auc, all_auc))

# lstm-xv3-bn-bi-alllayer, epoch 30: F_AUC: 0.9001, B_AUC: 0.9042, ALL_AUC: 0.9065
# lstm-xv3-dropout-bi-alllayer, epoch 30: F_AUC: 0.85, B_AUC: 0.85, ALL_AUC: 0.85
# lstm-on-category-dataset, F_AUC: 0.6497, B_AUC: 0.6635, ALL_AUC: 0.6611
# lsmt-on-category-dataset-old-model, F_AUC: 0.7498, B_AUC: 0.7394, ALL_AUC: 0.7492

# Fill in the blank test
criterion = nn.CrossEntropyLoss(reduction='none')
is_correct_f = []
is_correct_b = []
is_correct_all = []
for idx in tqdm.trange(len(test_dataset)):
    items, labels, question_part, question_id, options, option_labels= test_dataset.get_fitb_quesiton(idx)
    lengths = torch.tensor([len(labels)-1 for _ in range(4)]).to(device) # 4 options
    substitute_part = labels.index(question_id)

    images = [items]
    for option in options:
        new_outfit = items.clone()
        new_outfit[substitute_part] = option
        images.append(new_outfit)
    images = torch.cat(images).to(device)

    with torch.no_grad():
        emb_seqs = encoder_cnn(images)
        emb_seqs = emb_seqs.reshape((4, len(labels), 512))
    f_input_embs = emb_seqs[:, :-1, :] # (1, 2, 3, 4)
    b_input_embs = flip_tensor(emb_seqs[:, 1:, :], dim=1) # (5, 4, 3, 2)

    with torch.no_grad():
        f_output = f_rnn(f_input_embs, lengths)
        b_output = b_rnn(b_input_embs, lengths)

    f_score = torch.matmul(f_output, test_features_matrix.t())
    b_score = torch.matmul(b_output, test_features_matrix.t())

    # The order of targets in a batch should follow the rule of 
    # torch.nn.utils.rnn.pack_padded_sequence
    f_targets = []
    b_targets = []
    option_labels = [labels[substitute_part]] + option_labels
    for i in range(1, len(labels)):
        for j in range(4):
            if i == substitute_part:
                f_targets.append(option_labels[j])
            else:
                f_targets.append(labels[i])

    for i in range(len(labels)-2, -1, -1):
        for j in range(4):
            if i == substitute_part:
                b_targets.append(option_labels[j])
            else:
                b_targets.append(labels[i])
    # Transform id to index
    f_targets = [test_features_ids.index(i) for i in f_targets] # (2, 3, 4, 5) 
    b_targets = [test_features_ids.index(i) for i in b_targets] # (4, 3, 2, 1)

    f_loss = criterion(f_score, torch.tensor(f_targets).to(device))
    b_loss = criterion(b_score, torch.tensor(b_targets).to(device))
    f_loss = f_loss.reshape(-1, 4)
    b_loss = b_loss.reshape(-1, 4)
    f_loss = f_loss.sum(dim=0)
    b_loss = b_loss.sum(dim=0)
 
    if f_loss.argmin().item() == 0:
        is_correct_f.append(True)
    else:
        is_correct_f.append(False)

    if b_loss.argmin().item() == 0:
        is_correct_b.append(True)
    else:
        is_correct_b.append(False)

    all_loss = f_loss + b_loss
    if all_loss.argmin().item() == 0:
        is_correct_all.append(True)
    else:
        is_correct_all.append(False)

print("F_ACC: {:.4f}".format(sum(is_correct_f) / len(is_correct_f)))
print("B_ACC: {:.4f}".format(sum(is_correct_b) / len(is_correct_b)))
print("ALL_ACC: {:.4f}".format(sum(is_correct_all) / len(is_correct_all)))

"""
# Train with original dataset, test use after dataset
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2463/2463 [02:05<00:00, 19.56it/s]
F_AUC: 0.7535, B_AUC: 0.7552, ALL_AUC: 0.7592
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2463/2463 [04:22<00:00,  9.38it/s]
F_ACC: 0.4734
B_ACC: 0.4495
ALL_ACC: 0.4689


# Train with after dataset
100%|████████████████████████████████████████████████████████████| 2463/2463 [02:06<00:00, 19.42it/s]
F_AUC: 0.5079, B_AUC: 0.5018, ALL_AUC: 0.5078
100%|████████████████████████████████████████████████████████████| 2463/2463 [04:22<00:00,  9.40it/s]
F_ACC: 0.2184
B_ACC: 0.2452
ALL_ACC: 0.2355
"""
