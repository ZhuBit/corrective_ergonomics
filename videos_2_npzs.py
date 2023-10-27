import os
import numpy as np
from ergonomics import RULA
from utils import *



from hpe_wrapper import Wrapper_2Dpose, Wrapper_3Dpose


def process_videos(sourcedir, targetdir):
    Setup_environment()
    model_2D ='./detectron/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    weights_2D = './detectron/checkpoint/model_final_997cc7.pkl'
    model_3D = './VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'

    pose2d = Wrapper_2Dpose(model=model_2D, weights= weights_2D , ROI_thr=0.75)
    pose_3d = Wrapper_3Dpose(model_3D)
    # Path to the videos folder (specified by the sourcedir argument)
    videos_folder_path = sourcedir

    # Path to the npz folder (specified by the targetdir argument)
    npz_folder_path = targetdir

    # Ensure the npz folder exists or create it
    if not os.path.exists(npz_folder_path):
        os.makedirs(npz_folder_path)

    # Iterate through all files in the videos folder
    for video_file in os.listdir(videos_folder_path):
        # Check if the file is a video (e.g., has .mp4 extension, you can add more extensions if needed)
        if video_file.endswith('.mp4'):
            print(video_file)
            video_path = os.path.join(videos_folder_path, video_file)

            # Process the video
            video_object = Video_wrapper(video_path, resize_video_by=0.5, start=0, end=10) # downsample_fps_by=2,
            data_2d, metadata_vid = pose2d.predict_2D_poses(input_video_object=video_object)
            data_3d = pose_3d.predict_3D_poses(data_2d, metadata_vid)

            # Extract the original video name without extension
            video_name_without_extension = os.path.splitext(video_file)[0]

            # Save 3D data with dictionary structure to the target directory
            npz_3d_file_path = os.path.join(npz_folder_path, f"{video_name_without_extension}_3d.npz")
            np.savez(npz_3d_file_path, **{
                'kps': data_3d,
                'RULA_scores': scores
            })

# Example usage:
if __name__ == "__main__":
    sourcedir = './videos/Dataset'  # Replace with your source directory
    targetdir = './npz'  # Replace with your target directory
    process_videos(sourcedir, targetdir)
