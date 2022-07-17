from preprocessing.txt_to_json import convert_txt_to_json
from preprocessing.extract_frames import extract_frames_from_videos
from preprocessing.extract_features import extract_feats
from preprocessing.prepro_text import creat_vocab_preprocess_text
from opts import opts


data_root = opts['data_root']
frame_path = opts['frame_path']
feat_path = opts['feat_path']
n_frames = opts['n_frames']
device = opts['device']
model_name = opts['feat_model']


if __name__ == '__main__':
    convert_txt_to_json(dataRoot=data_root, txtName='caption.txt')
    extract_frames_from_videos(frameRoot=frame_path, feat_path=feat_path)
    extract_feats(frameRoot=frame_path, feat_path=feat_path, n_frames=n_frames, device=device, model_name=model_name)
    creat_vocab_preprocess_text(root=data_root, capName='caption.json')
