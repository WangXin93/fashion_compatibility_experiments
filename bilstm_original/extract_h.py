from model import GRUModel, EncoderCNN, LSTMModel
from polyvore_dataset import create_dataloader
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.nn as nn
import logging
import torch
import os
import sys
import pickle

batch_size = 32
comment = ''
model = 'lstm'
emb_size = 512

device = torch.device("cuda")
_, dataloader = create_dataloader(batch_size=batch_size, which_set='test', shuffle=False, img_size=299)

encoder_cnn = EncoderCNN(emb_size)
encoder_cnn.load_state_dict(torch.load('./encoder_cnn_vse.pth'))
encoder_cnn = encoder_cnn.to(device)
encoder_cnn.train(False)

test_features = {}
for batch_num, input_data in enumerate(dataloader, 1):
	print('#{}\r'.format(batch_num), end='', flush=True)
	lengths, names, likes, desc, images, image_ids = input_data

	image_seqs = images.to(device)
	with torch.no_grad():
		emb_seqs = encoder_cnn(image_seqs)

	batch_ids = []
	for outfit in image_ids:
		for i in outfit:
			batch_ids.append(i)

	for i, id in enumerate(batch_ids):
		test_features[id] = emb_seqs[i].cpu().detach().numpy()

pickle.dump(test_features, open('test_features.pkl', 'wb'))
print('Done.')
