import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vgg16, alexnet
from C3D import Path, C3D

import os
import tqdm
import numpy as np
from PIL import Image


def extract_feats(frame_path, feat_path, n_frames, device, model_name):
    '''
    extract features from video using pretrained CNN model
    Args:
        frame_path(str): path of frames
        feat_path(str): path to save features
        n_frames(int): number of frames used to extract features
        device(torch.device): device to compute data
        model_name(str): name of CNN used to extract features ('vgg16', 'alexnet', 'C3D')
    '''
    if model_name == 'vgg16':
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Identity()
    elif model_name == 'alexnet':
        model = alexnet(pretrained=True)
        model.classifier[6] = nn.Identity()
    else:
        model = C3D(pretrained=True)
    model.to(device)
    model.eval()

    if model_name in ['vgg16', 'alexnet']:
        transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((227, 227)),
            transforms.ToTensor()
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((227, 227)),
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    vids = os.listdir(frame_path)

    for vid in tqdm.tqdm(vids, desc='Extracting features'):
        frames = os.listdir(frame_path + vid)
        samples_idx = np.linspace(0, len(frames)-1, n_frames).astype(int)
        samples = [str(i) + '.jpg' for i in samples_idx]

        if model_name in ['vgg16', 'alexnet']:
            imgs = torch.ones(len(samples), 3, 227, 227)
        else:
            imgs = torch.ones(len(samples), 3, 112, 112)

        for i, item in enumerate(samples):
            img = transformer(Image.open(frame_path + vid + '/' + item))
            imgs[i] = img

        if model_name in ['vgg16', 'alexnet']:
            imgs = imgs.to(device)
        else:
            imgs = imgs.transpose(1, 0).unsqueeze(0).to(device)

        with torch.no_grad():
            if model_name in ['vgg16', 'alexnet']:
                feats = model(imgs).squeeze()
            else:
                feats = model(imgs)
        feats = feats.cpu().numpy()
        np.save(feat_path + vid + ".npy", feats)
