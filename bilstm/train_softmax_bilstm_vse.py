import argparse
import logging
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from model import EncoderCNN, LSTMModel
from polyvore_dataset_name import CategoryDataset, lstm_collate_fn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

################################# Argparse ####################################
parser = argparse.ArgumentParser(description="Polyvore BiLSTM")
parser.add_argument("--model", type=str, default="lstm")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--comment", type=str, default="")
args = parser.parse_args()
print(args)

###############################################################################

epochs = args.epochs
batch_size = args.batch_size
comment = args.comment
model = args.model
emb_size = 512
log_step = 2
device = torch.device("cuda")

################################# DataLoader ##################################
img_size = 299
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)
train_dataset = CategoryDataset(transform=transform,
                                data_file="train_no_dup_with_category_3more_name.json",
                                use_mean_img=False,
                                neg_samples=False)
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=lstm_collate_fn,
)
val_dataset = CategoryDataset(
    transform=transform,
    data_file="valid_no_dup_with_category_3more_name.json",
    use_mean_img=False,
    neg_samples=False
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=lstm_collate_fn
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
###############################################################################

encoder_cnn = EncoderCNN(emb_size, need_rep=True)
encoder_cnn = encoder_cnn.to(device)

if model == "lstm":
    f_rnn = LSTMModel(emb_size, emb_size, emb_size, device, bidirectional=False)
    b_rnn = LSTMModel(emb_size, emb_size, emb_size, device, bidirectional=False)
f_rnn = f_rnn.to(device)
b_rnn = b_rnn.to(device)

embedding = nn.Embedding(len(train_dataset.vocabulary), emb_size)
embedding = embedding.to(device)
image_embedding = nn.Linear(2048, emb_size)
image_embedding = image_embedding.to(device)

criterion = nn.CrossEntropyLoss()
params_to_train = (
    list(encoder_cnn.parameters())
    + list(f_rnn.parameters())
    + list(b_rnn.parameters())
    + list(embedding.parameters())
    + list(image_embedding.parameters())
)
optimizer = torch.optim.SGD(params_to_train, lr=2e-1, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

################################## Logger #####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("log_{}{}.log".format(__file__.split(".")[0], comment))
log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"
formatter = logging.Formatter(log_format)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)

################################## Train ######################################
def flip_tensor(tensor, device=device):
    """Flip a tensor in 0 dim for backward rnn.
    """
    idx = [i for i in range(tensor.size(0) - 1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    flipped_tensor = tensor.index_select(0, idx)
    return flipped_tensor

def vse_loss(semb, vemb, cap_mask, emb_size=emb_size, margin=0.2):
    """Visual Semantic Embedding loss"""
    semb = torch.masked_select(semb, cap_mask.unsqueeze(dim=1))
    vemb = torch.masked_select(vemb, cap_mask.unsqueeze(dim=1))
    semb = semb.reshape([-1, emb_size])
    vemb = vemb.reshape([-1, emb_size])
    scores = torch.matmul(semb, vemb.transpose(0, 1))
    diagnoal = scores.diag().unsqueeze(dim=1)
    cost_s = torch.clamp(margin - diagnoal + scores, min=0, max=1e6)  # 0.2 is margin
    cost_im = torch.clamp(margin - diagnoal.transpose(0, 1) + scores, min=0, max=1e6)
    cost_s = cost_s - torch.diag(cost_s.diag())
    cost_im = cost_im - torch.diag(cost_im.diag())
    emb_loss = cost_s.sum() + cost_im.sum()
    emb_loss = emb_loss / (semb.shape[0] ** 2)
    return emb_loss

def train():
    for epoch in range(1, epochs + 1):
        # Train phase
        total_loss = 0
        scheduler.step()
        encoder_cnn.train(True)
        f_rnn.train(True)
        b_rnn.train(True)
        for batch_num, input_data in enumerate(train_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = input_data
            image_seqs = images.to(device)  # (20+, 3, 224, 224)
            emb_seqs, rep = encoder_cnn(image_seqs)  # (20+, 512)

            # Encode Normalized Semantic Embedding
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
            # Encode normalized Visual Representations
            vemb = F.normalize(image_embedding(rep), dim=1)
            # VSE Loss
            emb_loss = vse_loss(semb, vemb, cap_mask, emb_size=emb_size, margin=0.2)

            # Generate input embeddings e.g. (1, 2, 3, 4)
            input_emb_list = []
            start = 0
            for length in lengths:
                input_emb_list.append(emb_seqs[start : start + length - 1])
                start += length
            f_input_embs = rnn_utils.pad_sequence(
                input_emb_list, batch_first=True
            )  # (4, 7, 512) (1, 2, 3, 4)
            b_target_embs = rnn_utils.pad_sequence(
                [flip_tensor(e) for e in input_emb_list], batch_first=True
            )  # (4, 3, 2, 1)

            # Generate target embeddings e.g. (2, 3, 4, 5)
            target_emb_list = []
            start = 0
            for length in lengths:
                target_emb_list.append(emb_seqs[start + 1 : start + length])
                start += length
            f_target_embs = rnn_utils.pad_sequence(
                target_emb_list, batch_first=True
            )  # (2, 3, 4, 5)
            b_input_embs = rnn_utils.pad_sequence(
                [flip_tensor(e) for e in target_emb_list], batch_first=True
            )  # (5, 4, 3, 2)

            seq_lengths = torch.tensor([i - 1 for i in lengths]).to(device)
            f_target_embs = rnn_utils.pack_padded_sequence(
                f_target_embs, seq_lengths, batch_first=True
            )[0]
            b_target_embs = rnn_utils.pack_padded_sequence(
                b_target_embs, seq_lengths, batch_first=True
            )[0]

            f_output = f_rnn(f_input_embs, seq_lengths)
            f_score = torch.matmul(f_output, f_target_embs.t())
            f_loss = criterion(f_score, torch.arange(f_score.shape[0]).to(device))
            b_output = b_rnn(b_input_embs, seq_lengths)
            b_score = torch.matmul(b_output, b_target_embs.t())
            b_loss = criterion(b_score, torch.arange(b_score.shape[0]).to(device))
            all_loss = f_loss + b_loss + 1. * emb_loss

            encoder_cnn.zero_grad()
            f_rnn.zero_grad()
            b_rnn.zero_grad()
            all_loss.backward()
            nn.utils.clip_grad_norm_(params_to_train, 0.5)  # clip gradient
            optimizer.step()

            total_loss += all_loss.item()
            # Print log info
            if batch_num % log_step == 0:
                logger.info(
                    "Epoch [{}/{}], Step #{}, F_loss: {:.4f}, B_loss: {:.4f}, VSE_Loss: {:.4f}, All_loss: {:.4f}".format(
                        epoch,
                        epochs,
                        batch_num,
                        f_loss.item(),
                        b_loss.item(),
                        emb_loss.item(),
                        all_loss.item(),
                    )
                )

        logger.info(
            "**Epoch {}**, Train Loss: {:.4f}".format(epoch, total_loss / batch_num)
        )
        # Save the model checkpoints
        torch.save(
            f_rnn.state_dict(), os.path.join("f_rnn{}.pth".format(comment))
        )
        torch.save(
            b_rnn.state_dict(), os.path.join("b_rnn{}.pth".format(comment))
        )
        torch.save(
            encoder_cnn.state_dict(),
            os.path.join("encoder_cnn{}.pth".format(comment)),
        )

        # Validate phase !!!
        encoder_cnn.train(False)  # eval mode (batchnorm uses moving mean/variance
        f_rnn.train(False)  # eval mode (batchnorm uses moving mean/variance
        b_rnn.train(False)  # eval mode (batchnorm uses moving mean/variance
        total_loss = 0
        for batch_num, input_data in enumerate(val_loader, 1):
            lengths, images, names, offsets, set_ids, labels, is_compat = input_data
            image_seqs = images.to(device)  # (20+, 3, 224, 224)
            with torch.no_grad():
                emb_seqs, _ = encoder_cnn(image_seqs)  # (20+, 512)

            # Generate input embeddings e.g. (1, 2, 3, 4)
            input_emb_list = []
            start = 0
            for length in lengths:
                input_emb_list.append(emb_seqs[start : start + length - 1])
                start += length
            f_input_embs = rnn_utils.pad_sequence(
                input_emb_list, batch_first=True
            )  # (4, 7, 512) (1, 2, 3, 4)
            b_target_embs = rnn_utils.pad_sequence(
                [flip_tensor(e) for e in input_emb_list], batch_first=True
            )  # (4, 3, 2, 1)

            # Generate target embeddings e.g. (2, 3, 4, 5)
            target_emb_list = []
            start = 0
            for length in lengths:
                target_emb_list.append(emb_seqs[start + 1 : start + length])
                start += length
            f_target_embs = rnn_utils.pad_sequence(
                target_emb_list, batch_first=True
            )  # (2, 3, 4, 5)
            b_input_embs = rnn_utils.pad_sequence(
                [flip_tensor(e) for e in target_emb_list], batch_first=True
            )  # (5, 4, 3, 2)

            seq_lengths = torch.tensor([i - 1 for i in lengths]).to(device)
            f_target_embs = rnn_utils.pack_padded_sequence(
                f_target_embs, seq_lengths, batch_first=True
            )[0]
            b_target_embs = rnn_utils.pack_padded_sequence(
                b_target_embs, seq_lengths, batch_first=True
            )[0]

            with torch.no_grad():
                f_output = f_rnn(f_input_embs, seq_lengths)
                f_score = torch.matmul(f_output, f_target_embs.t())
                f_loss = criterion(f_score, torch.arange(f_score.shape[0]).to(device))
                b_output = b_rnn(b_input_embs, seq_lengths)
                b_score = torch.matmul(b_output, b_target_embs.t())
                b_loss = criterion(b_score, torch.arange(b_score.shape[0]).to(device))
                all_loss = f_loss + b_loss

            total_loss += all_loss.item()

        logger.info(
            "**Epoch {}**, Valid Loss: {:.4f}".format(epoch, total_loss / batch_num)
        )


if __name__ == "__main__":
    train()
