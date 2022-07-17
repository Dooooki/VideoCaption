import cv2
import os
import tqdm


def extract_frames_from_videos(video_path, frame_path):
    vids = os.listdir(video_path)

    for vid in tqdm.tqdm(vids, desc='Extracting frames: '):
        videoToFrames(vid, video_path, frame_path)


def videoToFrames(vid, video_path, frame_path):
    '''
        extract every frame of a video and save as jpg file
        Args:
            video_path(str): path of videos
            frame_path(str): path to save frames
    '''
    frame_dir = frame_path + vid[:-4]
    os.mkdir(frame_dir)

    cap = cv2.VideoCapture(video_path + vid)
    ret, frame = cap.read()
    count = 0
    while ret:
        cv2.imwrite(frame_dir + '/' + str(count) + ".jpg", frame)
        ret, frame = cap.read()
        count += 1
