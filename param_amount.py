from model import S2VTModel
from torchvision.models import vgg16, alexnet
from preprocessing.C3D import C3D

import torch
from ptflops import get_model_complexity_info
from opts import opts


def get_feat_model_amount(name, input):
    '''
    calculate GFLOPs and Parameters of a model used to extract video features
    Args: name(str): name of a model
          input(tuple): shape of the model input
    Returns: macs(str): GFLPs
             params(str): Parameters
    '''
    if name == 'alexnet':
        model = alexnet(pretrained=True)
    elif name == 'vgg16':
        model = vgg16(pretrained=True)
    else:
        model = C3D(pretrained=True)

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input, as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
    return macs, params


def get_S2VT_amount():
    '''
    calculate GFLOPs and Parameters of S2VT
    Returns: macs(str): GFLPs
             params(str): Parameters
    '''
    model = S2VTModel(vocab_size=opts['vocab_size'], max_len=opts['max_len'],
                      dim_hidden=opts['hidden_dim'], dim_word=opts['word_dim'], mode='test',
                      dim_vid=opts['feat_dim'], sos_id=opts['sos_idx'])

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (20, 4096), as_strings=True,
                                                 print_per_layer_stat=False, verbose=True)
    return macs, params


if __name__ == '__main__':
    a_macs, a_params = get_feat_model_amount('alexnet', (3, 227, 227))
    v_macs, v_params = get_feat_model_amount('vgg16', (3, 227, 227))
    c_macs, c_params = get_feat_model_amount('c3d', (3, 16, 112, 112))
    s_macs, s_params = get_S2VT_amount()

    print('AlexNet GFLOPs:{}, Parameters:{}'.format(a_macs, a_params))
    print('Vgg16 GFLOPs:{}, Parameters:{}'.format(v_macs, v_params))
    print('C3D GFLOPs:{}, Parameters:{}'.format(c_macs, c_params))
    print('S2VT GFLOPs:{}, Parameters:{}'.format(s_macs, s_params))
