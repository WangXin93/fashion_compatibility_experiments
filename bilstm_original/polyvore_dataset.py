import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import csv
import gzip
import json
from PIL import Image
import os

class PolyvoreDataset(Dataset):
    """ Polyvore Dataset."""

    # Initialize your data, download, etc.
    def __init__(self,
                 which_set='train',
                 root_dir='/export/home/wangx/datasets/polyvore-dataset/images/',
                 transform=None):

        self.root_dir = root_dir
        self.transform = transform
        filename = '/export/home/wangx/datasets/polyvore-dataset/polyvore/{}_no_dup.json'.format(
            which_set)

        with open(filename) as f:
            self.data = json.load(f)

        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.word_to_idx)
        self.vocabulary.append('UNK')
        with open('data/final_word_dict.txt') as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.word_to_idx)
                    self.vocabulary.append(name)

    def __getitem__(self, index):
        names = []
        for e in self.data[index]['items']:
            name = []
            for word in e['name'].split():
                name.append(word)
            names.append(torch.LongTensor(self.str_to_idx(name)))

        images = []
        image_ids = []
        set_id = self.data[index]['set_id']
        for e in self.data[index]['items']:
            img_index = e['index']
            img_path = os.path.join(self.root_dir, str(set_id), str(img_index)) + '.jpg'
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            images.append(img)
            image_ids.append('{}_{}'.format(set_id, img_index))
        #images.append(torch.ones_like(images[0])) # append an zero image as end-of-set

        input_images = images[:8]
        input_images = torch.stack(input_images)

        likes = self.data[index]['likes']
        desc = self.data[index]['desc']

        return (
            len(names),       # lengths
            names[:8],        # names
            likes,            # likes
            desc,             # desc
            input_images,     # images 
            image_ids,        # image_ids
        )


    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK'] for w in name] 


def collate_fn(data):
    """Need custom a collate_fn for names and images"""
    data.sort(key=lambda x:x[0], reverse=True)
    lengths, names, likes, descs, images, image_ids = zip(*data)
    images = torch.cat(images)
    names = sum(names, [])
    return (
        lengths,
        names,
        likes,
        descs,
        images,
        image_ids
    )


def create_dataloader(batch_size=4,
                      shuffle=True,
                      num_workers=4,
                      which_set='train',
                      img_size=224):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    dataset = PolyvoreDataset(transform=transform,
                              which_set=which_set)
    return dataset, DataLoader(dataset,
                               batch_size=batch_size,
                               shuffle=shuffle,
                               collate_fn=collate_fn,
                               num_workers=num_workers)


class PredictCompatibilityDataset(Dataset):
    """Dataset for compatibility AUC"""
    def __init__(self, 
                 root_dir="/export/home/wangx/datasets/polyvore-dataset/images/",
                 transform=None):
        self.root_dir = root_dir
        self.transform = transform
        filename = "/export/home/wangx/code/pytorch-tutorial/tutorials/03-advanced/my_polyvore/data/fashion_compatibility_prediction.txt"
        with open(filename) as f:
            self.data = f.readlines()

    def __getitem__(self, index):
        label, *image_ids = self.data[index].strip().split()
        images = []
        if self.transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((299, 299)),
                torchvision.transforms.ToTensor()
            ])
        for item in image_ids:
            img_path = os.path.join(self.root_dir, item.split('_')[0], item.split('_')[1]) + '.jpg'
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            images.append(img)
        input_images = torch.stack(images)
        return input_images, image_ids, label

    def __len__(self):
        return len(self.data)


# Test the loader
if __name__ == "__main__":
    d, l = create_dataloader()
    batch = next(iter(l))
    #d = PredictCompatibilityDataset()
