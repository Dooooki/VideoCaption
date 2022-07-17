import json
import torch

info_path = './data/MSVD/info.json'


def get_vocab_size(info_path):
    info = json.load(open(info_path, 'r'))
    return len(info['vocab']['word_to_idx'])


def get_idx(info_path, key):
    info = json.load(open(info_path, 'r'))
    return info['vocab']['word_to_idx'][key]


opts = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'data_root': './data/MSVD/',
    'video_path': './data/MSVD/Video/',    # path of videos
    'frame_path': './data/MSVD/Frame/',    # path of frames
    'feat_path': './data/MSVD/features/',    # path of video features
    'n_frames': 20,    # num of frames extracted from video
    'feat_model': 'vgg16',  # model used to extract video features ('vgg16', 'alexnet', 'c3d')
    'info_path': info_path,
    'batch_size': 8,
    'vocab_size': get_vocab_size(info_path),
    'feat_dim': 4096,
    'hidden_dim': 1000,
    'word_dim': 512,
    'max_len': 15,  # max_len of label(dataloader)
    'sos_idx': get_idx(info_path, '<sos>'),
    'eos_idx': get_idx(info_path, '<eos>'),

    'epoch': 200,
    'lr': 1e-5,
    'lr_patience': 5,
    'save_freq': 10
}

if __name__ == '__main__':
    print(get_vocab_size(info_path))
    print('<sos>:', get_idx(info_path, '<sos>'))
    print('<eos>:', get_idx(info_path, '<eos>'))
