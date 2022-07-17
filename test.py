import json
import numpy as np

import torch

from opts import opts
from model import S2VTModel
from misc.utils import decode_sequence

import warnings

warnings.filterwarnings('ignore')

device = opts['device']
data_Root = opts['dataRoot']
feat_path = opts['feat_path']
info_path = opts['info_path']
max_len = opts['max_len']
batch_size = opts['batch_size']
vocab_size = opts['vocab_size']
hidden_dim = opts['hidden_dim']
word_dim = opts['word_dim']
feat_dim = opts['feat_dim']
sos_idx = opts['sos_idx']
eos_idx = opts['eos_idx']


def test(vid, model_path):
    '''
    generate a caption for a video
    Args: vid(str): id of a video from MSVD dataset
          model_path(str): name of the model used to genrate caption
    '''
    model = S2VTModel(vocab_size=vocab_size, max_len=max_len, dim_hidden=hidden_dim, dim_word=word_dim,
                      dim_vid=feat_dim, sos_id=sos_idx, n_layers=1, rnn_cell='lstm').to(device)
    model.load_state_dict(torch.load(model_path))

    info = json.load(open(info_path, 'r'))
    captions = json.load(open(data_Root + 'caption.json', 'r'))
    idx_to_word = info['vocab']['idx_to_word']

    feat = np.load(feat_path + vid + '.npy')
    feat = torch.tensor(feat, dtype=torch.float, device=device).unsqueeze(0).to(device)
    label = torch.ones((1, max_len))
    _, pred = model(feat, label, 'test')
    sents = decode_sequence(idx_to_word, pred, eos_idx)
    print(vid + ': ', sents[0])


if __name__ == '__main__':
    vid = '_0nX-El-ySo_83_93'
    model_path = 'Model/epoch100.pth'
    test(vid, model_path)