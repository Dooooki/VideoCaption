import json
from collections import Counter
import tqdm

from sklearn.model_selection import train_test_split


def build_vocab(counter, freq_thres):
    '''
    Build a vocab between word and index
    Args: conuter(collections.Counter()): a set of all words
    	  freq_thres(int): the threshold of bad word, if the frequency of a word lower than this, it will be considered as a bad word ('<unk>')
    Returns: word_to_idx(dict): a dictionary in which the key is a word and the value is its index in the vocab
    		 idx_to_word(dict): a dictionary in which the key is an index and the value is its corresponding word in the vocab
    '''
    word_freq = counter.most_common()
    word_to_idx = {'<unk>': 0}
    for idx, (word, freq) in enumerate(tqdm.tqdm(word_freq, desc='Building vocab'), start=1):
        if freq < freq_thres:
            continue
        word_to_idx[word] = int(idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    print('Word num: ', len(word_to_idx))

    return word_to_idx, idx_to_word


def preprocess_text(captions, root, freq_thres):
    '''
    Split the dataset into train set, validation set and test set; tokenize the captions; save information
	Args: captions(dict): captions dict loaded from 'captions.json'
	      root(str): path of data
		  freq_thres(int): the threshold of bad word, if the frequency of a word lower than this, it will be considered as a bad word ('<unk>'); default: 2
    '''
    all_info = {}

    split = {}
    vids = [vid for vid, _ in captions.items()]
    train, val_test = train_test_split(vids, test_size=0.4)
    val, test = train_test_split(val_test, test_size=0.5)
    split['train'] = train
    split['val'] = val
    split['test'] = test
    all_info['split'] = split

    data = {}
    counter = Counter()
    for vid, caps in captions.items():
        tokens = []
        for cap in caps:
            token = cap.split()
            token = ['<sos>'] + token + ['<eos>']
            counter.update(token)
            tokens.append(token)
        data[vid] = tokens

    vocab = {}
    word_to_idx, idx_to_word = build_vocab(counter, freq_thres)
    vocab['word_to_idx'] = word_to_idx
    vocab['idx_to_word'] = idx_to_word
    all_info['vocab'] = vocab

    for vid, tokens in tqdm.tqdm(data.items(), desc='Converting word to index'):
        idxs = []
        for token in tokens:
            idx = [word_to_idx[word] if word in word_to_idx.keys() else word_to_idx['<unk>'] for word in token]
            idxs.append(idx)
        data[vid] = idxs
    all_info['data'] = data

    with open(root + 'info.json', 'w') as j:
        j.write(json.dumps(all_info))


def creat_vocab_preprocess_text(root, capName, freq_thres=2):
    capPath = root + capName
    captions = json.load(open(capPath, 'r'))
    preprocess_text(captions, root, freq_thres)
