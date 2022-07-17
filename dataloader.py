import json
import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from opts import opts


device = opts['device']


class VideoDataset(Dataset):
    def __init__(self, feat_path, info_path, mode, n_words, eos_idx):
        '''
        Args: feat_path(str): the path of the feats extracted from video
    		  info_path(str): the path of info.json
    		  mode(str): load corresponding dataset ('train', 'val', 'test')
    		  n_words(int): the max length of the captions
    		  eos_idx(int): the index of <eos>
        '''
        self.feat_path = feat_path
        self.feat_names = os.listdir(feat_path)

        info = json.load(open(info_path, 'r'))
        self.word_to_idx = info['vocab']['word_to_idx']
        self.idx_to_word = info['vocab']['idx_to_word']
        self.data = info['data']
        self.split = info['split']
        self.mode = mode
        self.n_words = n_words
        self.eos_idx = eos_idx
        # print('Loading {} set, nums of data: {}'.format(self.mode, len(self.split[self.mode])))

    def __getitem__(self, index):
        data = {}

        vids = self.split[self.mode]
        vid = vids[index]
        data['vid'] = vid

        feat = np.load(self.feat_path + vid + '.npy')
        feat = torch.tensor(feat, dtype=torch.float, device=device, requires_grad=True)
        data['feat'] = feat

        labels = self.data[vid]
        label = np.random.choice(labels, 1)[0]
        if len(label) > self.n_words:
            label = label[:self.n_words]
            label[-1] = self.eos_idx
        pad_label = torch.zeros(self.n_words, dtype=torch.long)
        pad_label[:len(label)] = torch.tensor(label, dtype=torch.long)
        data['label'] = pad_label

        mask = torch.zeros(self.n_words, dtype=torch.float)
        mask[:len(label)-1] = 1
        data['mask'] = mask

        return data

    def __len__(self):
        return len(self.split[self.mode])


if __name__ == '__main__':
    train_set = VideoDataset(feat_path=opts['feat_path'], info_path=opts['info_path'],
                             mode='train', n_words=opts['max_len'], eos_idx=opts['eos_idx'])
    train_loader = DataLoader(train_set, batch_size=opts['batch_size'], shuffle=True)

    data = next(iter(train_loader))
    vid = data['vid']
    feat = data['feat']
    label = data['label']
    mask = data['mask']
    print('vid:', len(vid))
    print('feat:', feat.shape)
    print('label:', label.shape)
    print('mask:', mask.shape)
