#import tkinter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
#import cv2
import numpy as np
import subprocess as sp
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
#from common.visualization import read_video, get_fps, get_resolution # reimplement so compatible with other models



import matplotlib
#matplotlib.use('Qt5Agg')
#matplotlib.use('TkAgg')

class heatmap_generator():
    def __init__(self, video_object, background_style='mean'):
        # self.path = path
        self.video_frames = video_object.frames
        if len(self.video_frames) > 100: #avoid RAM issues with loading too much memory
            step = len(self.video_frames) // 100
            self.video_frames = self.video_frames[::step]

        if background_style == 'mean':
            self.background = self._get_mean()
        elif background_style == 'median':
            self.background = self._get_med()
        else:
            self.background = self.video_frames[0]

        self.keypoints = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                      "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
                      "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14,
                      "LAnkle": 15, "RAnkle": 16}


    # def _load_vid(self, downsample_factor):
    #     all_frames = []
    #     for frame_i, im in enumerate(self._read_video()):  # skip=input_video_skip, limit=limit
    #         if frame_i % downsample_factor != 1 and downsample_factor != 1:
    #             continue
    #         all_frames.append(im[0]) #[...,::-1].copy()
    #     return all_frames
    #
    # def _read_video(self):
    #     def get_resolution(filename):
    #         command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
    #                    '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    #         with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
    #             for line in pipe.stdout:
    #                 w, h = line.decode().strip().split(',')
    #                 return int(w), int(h)
    #
    #     w, h = get_resolution(self.path)
    #     command = ['ffmpeg',
    #                '-i', self.path,
    #                '-f', 'image2pipe',
    #                '-pix_fmt', 'bgr24',
    #                '-vsync', '0',
    #                '-loglevel', 'quiet',
    #                '-vcodec', 'rawvideo', '-']
    #
    #     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    #     i = 0
    #     while True:
    #         i += 1
    #         data = pipe.stdout.read(w * h * 3)
    #         if not data:
    #             break
    #         yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3)), str(i - 1).zfill(5)

    def _transparent_color_map(self):
        color_array = plt.get_cmap('jet')
        nbins = color_array.N
        color_array = plt.get_cmap('jet')(range(nbins))
        first_half = np.linspace(0.0, 0.6, nbins//2)
        sec_half = first_half = np.linspace(0.6, 1.0, nbins-(nbins//2))
        alpha_sec = np.concatenate((first_half, sec_half), axis=0)
        color_array[:, -1] = alpha_sec
        map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name='rainbow_alpha', colors=color_array)
        return map_object


    def _get_med(self):
        frames = np.asarray(self.video_frames)
        # if len(frames) > 100: #take only 100 to avoid RAM issues
        #     step = len(frames)//100
        #     frames = frames[::step]
        #     med = np.median(frames[::step], axis=0)
        # else:
        med = np.median(frames, axis=0) # Take the median over the first dim
        return med

    def _get_mean(self):
        frames = np.asarray(self.video_frames)
        # if len(frames) > 100:
        #     step = len(frames)//100
        #     med = np.mean(frames[::step], axis=0)
        # else:
        med = np.mean(frames, axis=0) # Take the median over the first dim
        return med

    def print_keypoints(self):
        print(self.keypoints)


    def movement_heatmap(self, poses_2d, keypoint, map_threshold=0):
        image_size = self.background.shape

        background = self.background.astype(int)

        kps = poses_2d['keypoints']['data_2d']['custom']

        if keypoint == 'Mid':
            Rfoot = self.keypoints['RAnkle']
            Lfoot = self.keypoints['LAnkle']
        else:
            joint = self.keypoints[keypoint]

        xcoord = []
        ycoord = []

        for pose_in_frame in kps:  # alle Keypoints von allen Personen in 2 Vektoren zusammenfassen
            if keypoint == 'Mid':
                x_m = (pose_in_frame[Rfoot][0] + pose_in_frame[Lfoot][0]) / 2
                y_m = (pose_in_frame[Rfoot][1] + pose_in_frame[Lfoot][1]) / 2
                xcoord.append(x_m)  # *image_size[0]
                ycoord.append(y_m)
            else:
                xcoord.append(pose_in_frame[joint][0]) #*image_size[0]
                ycoord.append(pose_in_frame[joint][1]) #*image_size[1]


        # plt.hist2d(xcoord, ycoord, bins=[np.arange(0,image_size[1],50),np.arange(0,image_size[0],50)], alpha=0.5, cmap=plt.cm.rainbow)
        # plt.gca().set_aspect('equal')
        # plt.colorbar()
        # background_img = cv2.imread(PROJECT_PATH + '/data/images/background.jpg')
        # background_img = cv2.flip(background_img, 0)
        # background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(background_img, zorder=0)
        # plt.show()

        # fit an array of size [Ndim, Nsamples]
        data = np.vstack([xcoord, ycoord])
        kde = gaussian_kde(data)

        # evaluate on a regular grid
        xgrid = np.linspace(0, image_size[1], int(image_size[1] / 5))
        ygrid = np.linspace(0, image_size[0], int(image_size[0] / 5))
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Zgrid = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Zgrid /= Zgrid.max()

        #hist
        #H, xedges, yedges = np.histogram2d(xcoord, ycoord, bins=(xgrid, ygrid))
        #H /= H.max()

        #from scipy import ndimage
        #masked_Zgrid = ndimage.median_filter(Zgrid, size=50)
        masked_Zgrid = np.ma.masked_where(Zgrid < map_threshold, Zgrid)


        # # Show Background Image
        #backgroun = np.flip(self.background)
        # background_img = cv2.flip(self.background, 0)
        # background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

        # Plot the result as an image
        plt.figure(figsize=(16, 12), dpi=80)
        plt.imshow(np.fliplr(np.flip(background))) #, zorder=0
        plt.imshow(np.fliplr(np.flip(masked_Zgrid.reshape(Xgrid.shape))), #np.flip(
                   origin='lower', interpolation='bilinear',
                   vmin=map_threshold, vmax=1,
                   extent=[0, image_size[1], 0, image_size[0]],
                   cmap=self._transparent_color_map(), alpha=.5) #plt.cm.rainbow #viridis YlOrBr plt.cm.jet
        plt.title('Movement Heatmap\nOccurance of Keypoint - ' + str(keypoint))
        plt.axis('off')
        #plt.colorbar()
        #plt.savefig(DRIVE_PATH + '/Häufigkeit_Keypoint' + str(joint) + '.png')
        plt.show()

    def ergonomic_heatmap(self, poses_2d, keypoint, ergonomic_results, min_score=5, map_threshold=0):
        image_size = self.background.shape

        background = self.background.astype(int)

        kps = poses_2d['keypoints']['data_2d']['custom']

        if keypoint == 'Mid':
            Rfoot = self.keypoints['RAnkle']
            Lfoot = self.keypoints['LAnkle']
        else:
            joint = self.keypoints[keypoint]

        xcoord = []
        ycoord = []

        for i, pose_in_frame in enumerate(kps):  # alle Keypoints von allen Personen in 2 Vektoren zusammenfassen
            if ergonomic_results[i] < min_score: #skip if lower than minimal score from user
                continue

            if keypoint == 'Mid':
                x_m = (pose_in_frame[Rfoot][0] + pose_in_frame[Lfoot][0]) / 2
                y_m = (pose_in_frame[Rfoot][1] + pose_in_frame[Lfoot][1]) / 2
                xcoord.append(x_m)  # *image_size[0]
                ycoord.append(y_m)
            else:
                xcoord.append(pose_in_frame[joint][0]) #*image_size[0]
                ycoord.append(pose_in_frame[joint][1]) #*image_size[1]


        # plt.hist2d(xcoord, ycoord, bins=[np.arange(0,image_size[1],50),np.arange(0,image_size[0],50)], alpha=0.5, cmap=plt.cm.rainbow)
        # plt.gca().set_aspect('equal')
        # plt.colorbar()
        # background_img = cv2.imread(PROJECT_PATH + '/data/images/background.jpg')
        # background_img = cv2.flip(background_img, 0)
        # background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)
        # plt.imshow(background_img, zorder=0)
        # plt.show()

        # fit an array of size [Ndim, Nsamples]
        if len(xcoord) != 0:
            data = np.vstack([xcoord, ycoord])
            kde = gaussian_kde(data)
        else:
            print("No occurence of poses above the specified minimum score")
            return

        # evaluate on a regular grid
        xgrid = np.linspace(0, image_size[1], int(image_size[1] / 5))
        ygrid = np.linspace(0, image_size[0], int(image_size[0] / 5))
        Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
        Zgrid = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
        Zgrid /= Zgrid.max()

        #hist
        #H, xedges, yedges = np.histogram2d(xcoord, ycoord, bins=(xgrid, ygrid))
        #H /= H.max()

        #from scipy import ndimage
        #masked_Zgrid = ndimage.median_filter(Zgrid, size=50)
        masked_Zgrid = np.ma.masked_where(Zgrid < map_threshold, Zgrid)


        # # Show Background Image
        #backgroun = np.flip(self.background)
        # background_img = cv2.flip(self.background, 0)
        # background_img = cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)

        # Plot the result as an image
        plt.figure(figsize=(16, 12), dpi=80)
        plt.imshow(np.fliplr(np.flip(background))) #, zorder=0
        plt.imshow(np.fliplr(np.flip(masked_Zgrid.reshape(Xgrid.shape))), #np.flip(
                   origin='lower', interpolation='bilinear',
                   vmin=map_threshold, vmax=1,
                   extent=[0, image_size[1], 0, image_size[0]],
                   cmap=self._transparent_color_map(), alpha=.5) #plt.cm.rainbow #viridis YlOrBr plt.cm.jet
        plt.title('Ergonomic Heatmap\nOccurance of Keypoint - ' + str(keypoint))
        plt.axis('off')
        #plt.colorbar()
        #plt.savefig(DRIVE_PATH + '/Häufigkeit_Keypoint' + str(joint) + '.png')
        plt.show()

