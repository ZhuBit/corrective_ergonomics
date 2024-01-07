import torch
import sys
import os
from pathlib import Path
import subprocess as sp
import numpy as np
import cv2
import json

def Setup_environment():
    file_path = os.path.dirname(os.path.realpath(__file__))

    detectron_path = Path(file_path + '/detectron')
    pose3d_path = Path(file_path + '/VideoPose3D')
    #pose3d_path = Path(file_path + '/VideoPose3D/common')
    sort_path = Path(file_path + '/OC_SORT')

    sys.path.append(str(detectron_path))
    sys.path.append(str(pose3d_path))
    sys.path.append(str(sort_path))

    import warnings
    warnings.filterwarnings("ignore")


class Video_wrapper():
    def __init__(self, path, resize_video_by=1, downsample_fps_by=1, start=None, end=None, num_threads=1):
        """
        :param path: path to the video file
        :param resize_video_by: resizing factor for the video resolution
        :param downsample_fps_by: factor for reducing fps or
        :param start: cutting previous parts of video before value determined by user (in seconds)
        :param end: cutting the ending of the video after value determined by end (in seconds)
        :param num_threads: number of threads for processing
        """
        self.path = path
        self.resize = resize_video_by
        self.downsample = int(max(downsample_fps_by, 1))

        self.start = start
        self.end = end

        self.threads = num_threads

        self.w, self.h, self.fps, self.dur = self._get_video_information()

        if self.end is not None:
            if self.end > self.dur:
                self.end = None
            else:
                self.dur -= self.end

        if self.start is not None:
            if self.start < 0:
                self.start = None
            else:
                self.dur -= self.start

        if self.downsample != 1:
            self.fps = int(self.fps / self.downsample)

        self.frames = self.load_video()


    def load_video(self):
        all_frames = []
        for frame_i, im in enumerate(self._read_video()):  # skip=input_video_skip, limit=limit
            if self.start is not None and frame_i < (self.start*int(self.fps*self.downsample)):
                continue
            if self.end is not None and frame_i > (self.end*int(self.fps*self.downsample)):
                continue

            if frame_i % self.downsample != 1 and self.downsample != 1:
                continue

            all_frames.append(im[0]) #[...,::-1].copy()
        return all_frames

    def _get_video_information(self):
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'csv=p=0', self.path]
        with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
            for line in pipe.stdout:
                w, h, fps, dur = line.decode().strip().split(',')
                fps = fps.split('/')
                fps = int(fps[0]) / int(fps[1])
                return int(w), int(h), int(fps), float(dur)

    # def _get_fps(self):
    #     command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
    #                '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', self.path]
    #     with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
    #         for line in pipe.stdout:
    #             a, b = line.decode().strip().split('/')
    #             return (int(a) / int(b)) / self.downsample

    def _read_video(self):
        # def get_resolution(filename):
        #     command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        #                '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
        #     with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        #         for line in pipe.stdout:
        #             w, h = line.decode().strip().split(',')
        #             return int(w), int(h)

        w, h, _, _ = self._get_video_information()
        if self.resize == 1:
            command = ['ffmpeg',
                       '-i', self.path,
                       '-f', 'image2pipe',
                       '-pix_fmt', 'bgr24',
                       '-vsync', '0',
                       '-loglevel', 'quiet',
                       '-vcodec', 'rawvideo', '-']
        else:
            w = int(w * self.resize)
            self.w = w
            h = int(h * self.resize)
            self.h = h

            command = ['ffmpeg', '-y',
                       '-i', self.path,
                       '-threads', str(self.threads),
                       '-vf','scale=%d:%d' % (w, h),
                       '-f', 'image2pipe',
                       '-pix_fmt', 'bgr24',
                       '-vsync', '0',
                       '-loglevel', 'quiet',
                       '-vcodec', 'rawvideo', '-']

        pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
        i = 0
        while True:
            i += 1
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3)), str(i - 1).zfill(5)


def pack_3D_keypoints(keypoints, boxes):#np arr(num frames, 17, 3), np arr(num frames, 17, 2), num_framses*5(5 val for boxes
    #media pipe???
    # har on skeleton data?

    return {
            'start_frame': 0,  # Inclusive
            'end_frame': len(keypoints),  # Exclusive
            'bounding_boxes': boxes,  # boxes
            'keypoints': keypoints,  # keypoints
        }

def videos_2_keypoints():
    # Setting up directories
    sourcedir = 'data/videos'
    targetdir = 'data/npz'

    # Setup the environment and models (assuming Setup_environment, Wrapper_2Dpose, and Wrapper_3Dpose are defined elsewhere)
    Setup_environment()
    model_2D ='./detectron/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    weights_2D = './detectron/checkpoint/model_final_997cc7.pkl'
    model_3D = './VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'

    pose2d = Wrapper_2Dpose(model=model_2D, weights=weights_2D, ROI_thr=0.75)
    pose_3d = Wrapper_3Dpose(model_3D)

    # Check if the target directory exists, if not create it
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Process each video in the source directory
    for video_file in os.listdir(sourcedir):
        if video_file.endswith('.mp4'):
            print(f"Processing {video_file}...")
            video_path = os.path.join(sourcedir, video_file)

            # Video processing (assuming Video_wrapper, Wrapper_2Dpose.predict_2D_poses, Wrapper_3Dpose.predict_3D_poses are defined)
            video_object = Video_wrapper(video_path, resize_video_by=0.5, start=0, end=10)  # Example parameters
            data_2d, metadata_vid = pose2d.predict_2D_poses(input_video_object=video_object)
            data_3d = pose_3d.predict_3D_poses(data_2d, metadata_vid)

            # Save the 3D data to a .npz file
            video_name_without_extension = os.path.splitext(video_file)[0]
            npz_3d_file_path = os.path.join(targetdir, f"{video_name_without_extension}_3d.npz")
            np.savez(npz_3d_file_path, **{'kps': data_3d, 'RULA_scores': scores})


