import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):

    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.loss_fn = nn.NLLLoss(reduce=False)

    def forward(self, logits, target, mask):
        """
        logits: shape of (N, seq_len, vocab_size)
        target: shape of (N, seq_len)
        mask: shape of (N, seq_len)
        """
        # truncate to the same size
        batch_size = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss_fn(logits, target)
        output = torch.sum(loss * mask) / batch_size
        return output


def decode_sequence(idx_to_word, preds, eos_idx):
    preds = preds.cpu()
    batch_size, max_len = preds.size()
    out = []
    for pred in preds:
        sent = ''
        for i in range(max_len):
            idx = int(pred[i].item())
            if idx == eos_idx:
                break
            else:
                sent += (idx_to_word[str(idx)] if i==0 else ' '+idx_to_word[str(idx)])
        out.append(sent)

    return out
