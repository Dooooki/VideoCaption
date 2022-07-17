import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.nn.utils import clip_grad_value_

from opts import opts
from model import S2VTModel
from dataloader import VideoDataset
from misc.utils import LanguageModelCriterion

from tensorboardX import SummaryWriter


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

log_path = 'logs'
model_path = './Model/epoch'


def train():
    writer = SummaryWriter(log_path)

    train_set = VideoDataset(info_path=info_path, feat_path=feat_path, mode='train', n_words=max_len, eos_idx=eos_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = VideoDataset(info_path=info_path, feat_path=feat_path, mode='val', n_words=max_len, eos_idx=eos_idx)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model = S2VTModel(vocab_size=vocab_size, max_len=max_len, dim_hidden=hidden_dim, dim_word=word_dim,
                      dim_vid=feat_dim, sos_id=sos_idx, n_layers=1, rnn_cell='lstm').to(device)
    criterion = LanguageModelCriterion()
    optimizer = optim.Adam(model.parameters(), lr=opts['lr'])
    # optimizer = optim.SGD(model.parameters(), lr=opts['lr'], momentum=0.8)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    for epoch in range(1, opts['epoch']+1):
        lr_scheduler.step()
        train_loss = 0.0
        model.train()
        for data in train_loader:
            feats = data['feat'].to(device)
            labels = data['label'].to(device)
            masks = data['mask'].to(device)

            optimizer.zero_grad()
            seq_probs, _ = model(feats, labels, 'train')
            loss = criterion(seq_probs, labels[:, 1:], masks[:, 1:])
            loss.backward()
            clip_grad_value_(model.parameters(), 5)
            optimizer.step()
            train_loss += loss.item()

        writer.add_scalar("train loss", train_loss, epoch)

        val_loss = 0.0
        model.eval()
        for data in val_loader:
            feats = data['feat'].to(device)
            labels = data['label'].to(device)
            masks = data['mask'].to(device)
            with torch.no_grad():
                seq_probs, _ = model(feats, labels, 'train')
                loss = criterion(seq_probs, labels[:, 1:], masks[:, 1:])
            val_loss += loss.item()
        writer.add_scalar("val loss", val_loss, epoch)

        if epoch % 10 == 0:
            path = model_path+str(epoch)+'.pth'
            torch.save(model.state_dict(), path)

        print('epoch:{}  train_loss:{}, val_loss:{}'.format(epoch, train_loss, val_loss))


if __name__ == '__main__':
    train()
