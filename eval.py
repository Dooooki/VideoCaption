import json
from itertools import chain

import torch
from torch.utils.data import DataLoader

from opts import opts
from model import S2VTModel
from dataloader import VideoDataset
from misc.utils import decode_sequence
from misc.cocoeval import COCOScorer

device = opts['device']
feat_path = opts['feat_path']
info_path = opts['info_path']
max_len = opts['max_len']   # max_len of label in dataloder
batch_size = opts['batch_size']
vocab_size = opts['vocab_size']
hidden_dim = opts['hidden_dim']
word_dim = opts['word_dim']
feat_dim = opts['feat_dim']
sos_idx = opts['sos_idx']
eos_idx = opts['eos_idx']
model_path = './Model_vgg16_lstm_20frames_e-5/epoch80.pth'


def convert_data_to_coco_scorer_format(data, idx_to_word):
    '''
    convert sequence of index to sentences
    Args: data(dict): captions dict from info.json
          idx_to_word(dict): index_to_word vocabulary from info.json
    Returns: out(dirct): containing video id and corresponding captions
    '''
    out = {}
    for id, caps in data.items():
        for cap in caps:
            sent = [idx_to_word[str(idx)] for idx in cap[1:-1]]
            sent = ' '.join(sent)
            if id in out.keys():
                out[id].append({'image_id': id, 'cap_id': len(out[id]), 'caption': sent})
            else:
                out[id] = []
                out[id].append({'image_id': id, 'cap_id': len(out[id]), 'caption': sent})
    return out


def eval():
    train_set = VideoDataset(info_path=info_path, feat_path=feat_path, mode='train', n_words=max_len, eos_idx=eos_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = VideoDataset(info_path=info_path, feat_path=feat_path, mode='val', n_words=max_len, eos_idx=eos_idx)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_set = VideoDataset(info_path=info_path, feat_path=feat_path, mode='test', n_words=max_len, eos_idx=eos_idx)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    msvd_loader = chain(train_loader, val_loader, test_loader)

    model = S2VTModel(vocab_size=vocab_size, max_len=max_len, dim_hidden=hidden_dim, dim_word=word_dim,
                      dim_vid=feat_dim, sos_id=sos_idx, n_layers=1, rnn_cell='lstm').to(device)
    model.load_state_dict(torch.load(model_path))
    scorer = COCOScorer()

    info = json.load(open(info_path, 'r'))
    idx_to_word = info['vocab']['idx_to_word']
    gts = convert_data_to_coco_scorer_format(info['data'], idx_to_word)

    samples = {}
    model.eval()
    print('Predicting...')
    for data in msvd_loader:
        feats = data['feat'].to(device)
        labels = data['label'].to(device)
        vids = data['vid']
        with torch.no_grad():
            seq_probs, seq_preds = model(feats, labels, 'test')

        sents = decode_sequence(idx_to_word, seq_preds, eos_idx)

        for i, vid in enumerate(vids):
            samples[vid] = [{'image_id': vid, 'caption': sents[i]}]

    print('Evaluating...')
    scores = scorer.score(gts, samples, samples.keys())
    for method, score in scores.items():
        print(method+': ', score)


if __name__ == '__main__':
    eval()
