import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
from common.skeleton import Skeleton

# from common.visualization import read_video, get_fps, get_resolution

# def downsample_tensor(X, factor):
#     length = X.shape[0]//factor * factor
#     return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)
#
# def read_video_test(filename):
#     w, h = get_resolution(filename)
#
#     command = ['ffmpeg',
#                '-i', filename,
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
#
#
# def read_video_custom(filename):
#     w, h = get_resolution(filename)
#
#     command = ['ffmpeg',
#                '-i', filename,
#                '-f', 'image2pipe',
#                '-pix_fmt', 'bgr24',
#                '-vsync', '0',
#                '-loglevel', 'quiet',
#                '-vcodec', 'rawvideo', '-']
#
#     pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
#     while True:
#         data = pipe.stdout.read(w * h * 3)
#         if not data:
#             break
#         yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))

#
# def render_animation_test(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
#                           limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
#     """
#     TODO
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     plt.ioff()
#     fig = plt.figure(figsize=(size * (1 + len(poses)), size))
#     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
#     ax_in.get_xaxis().set_visible(False)
#     ax_in.get_yaxis().set_visible(False)
#     ax_in.set_axis_off()
#     ax_in.set_title('Input')
#
#     ax_3d = []
#     lines_3d = []
#     trajectories = []
#     radius = 1.7
#     for index, (title, data) in enumerate(poses.items()):
#         ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         ax.set_xlim3d([-radius/2, radius/2])
#         ax.set_zlim3d([0, radius])
#         ax.set_ylim3d([-radius/2, radius/2])
#         try:
#             ax.set_aspect('equal')
#         except NotImplementedError:
#             ax.set_aspect('auto')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 7.5
#         ax.set_title(title)  # , pad=35
#         ax_3d.append(ax)
#         lines_3d.append([])
#         trajectories.append(data[:, 0, [0, 1]])
#     poses = list(poses.values())
#
#     # # Decode video
#     # if input_video_path is None:
#     #     # Black background
#     #     all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
#     # else:
#     #     # Load video using ffmpeg
#     #     all_frames = []
#     #     for frame_i, im in enumerate(read_video(input_video_path)):  # skip=input_video_skip, limit=limit
#     #         all_frames.append(im[0])
#     #     #effective_length = min(keypoints.shape[0], len(all_frames))
#     #     #all_frames = all_frames[:effective_length]
#     #
#     #     keypoints = keypoints[input_video_skip:]  # todo remove
#     #     for idx in range(len(poses)):
#     #         poses[idx] = poses[idx][input_video_skip:]
#     #
#     #     if fps is None:
#     #         fps = get_fps(input_video_path)
#
#     # if downsample > 1:
#     #     #keypoints = downsample_tensor(keypoints, downsample)
#     #     #all_frames = all_frames[::downsample]
#     #     all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
#     #     # for idx in range(len(poses)):
#     #     #     poses[idx] = downsample_tensor(poses[idx], downsample)
#     #     #     trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
#     #     fps /= downsample
#
#     initialized = False
#     image = None
#     lines = []
#     points = None
#
#     # print("length frames")
#     # print(len(all_frames))
#     # print("limit is:")
#     # print(all_frames[0].shape)
#
#     if limit < 1:
#         limit = len(all_frames)
#     else:
#         limit = min(limit, len(all_frames))
#
#     parents = skeleton.parents()  # [-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12, 16]
#
#     def update_video(i):
#         nonlocal initialized, image, lines, points
#
#         for n, ax in enumerate(ax_3d):
#             ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#             ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
#         # print(n, ax)
#         # print(radius)
#         # print(trajectories[n][i, 0])
#         # print(i)
#
#         # print([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#
#         # ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
#         # ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])
#
#         # Update 2D poses
#         joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
#         colors_2d = np.full(17, 'black')  # keypoints.shape[1]
#         colors_2d[joints_right_2d] = 'red'
#
#         # print(colors_2d)
#         # print(keypoints.shape)
#
#         if not initialized:
#             print(all_frames[i].shape)
#             image = ax_in.imshow(all_frames[i], aspect='equal')
#
#             for j, j_parent in enumerate(parents):
#
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
#                     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
#
#                 col = 'red' if j in skeleton.joints_right() else 'black'
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
#                                                [pos[j, 1], pos[j_parent, 1]],
#                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
#
#             points = ax_in.scatter(*keypoints[i].T, 10, edgecolors='white', zorder=10)  # color=colors_2d,
#
#             initialized = True
#         else:
#             image.set_data(all_frames[i])
#
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
#
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]  # [n][i]
#
#                     # print(pos.shape)
#
#                     lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
#                     lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
#                     lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
#
#             points.set_offsets(keypoints[i])
#
#         print('{}/{}      '.format(i, limit), end='\r')
#
#     fig.tight_layout()
#
#     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
#     if output.endswith('.mp4'):
#         Writer = writers['ffmpeg']
#         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#         anim.save(output, writer=writer)
#     elif output.endswith('.gif'):
#         anim.save(output, dpi=80, writer='imagemagick')
#     else:
#         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#     plt.close()


def render_animation_custom(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, size=6, input_video=None, input_video_skip=0):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """
    plt.ioff()
    fig = plt.figure(figsize=(size * (1 + len(poses)), size))

    if input_video is None:
        render_2D = False
    else:
        render_2D = True

        ax_in = fig.add_subplot(1, 1 + len(poses), 1)
        ax_in.get_xaxis().set_visible(False)
        ax_in.get_yaxis().set_visible(False)
        ax_in.set_axis_off()
        ax_in.set_title('2D Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        if render_2D == True:
            ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
        else:
            ax = fig.add_subplot(1, 1 + len(poses), (index + 1, index + 2), projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            ax.set_aspect('auto')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    poses = list(poses.values())

    # Decode video
    if input_video is None:
        fps = keypoints_metadata['video_metadata']['data_2d']['fps']

        # Black background
        #all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
        #render_2D = False
    else:
        fps = input_video.fps

    # if downsample > 1:
    #     #keypoints = downsample_tensor(keypoints, downsample)
    #     if render_2D == True:
    #         all_frames = all_frames[::downsample]
    #     #all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8') #memory issues as the whole video has to be loaded to RAM...
    #     #for idx in range(len(poses)):
    #     #    poses[idx] = downsample_tensor(poses[idx], downsample)
    #     #    trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
    #     fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None

    if limit < 1:
        limit = poses[0].shape[0]#len(all_frames)
    else:
        limit = min(limit, poses[0].shape[0])#len(all_frames))

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])

        # Update 2D poses
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            #render 2D video
            if render_2D:
                image = ax_in.imshow(input_video.frames[i][...,[2,1,0]], aspect='equal')
                points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            ###### ///////////////////////////////////////////////////////////
            # ax.scatter(poses[0][0, 1, 0], poses[0][0, 1, 1], 0, c='r', marker='o') #(255,0,0) poses[0][0, 1, 2]
            # ax.scatter(poses[0][0, 4, 0], poses[0][0, 4, 1], 0, c='r', marker='o') #poses[0][0, 4, 2]

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))


            initialized = True
        else:
            # render 2D video
            if render_2D:
                image.set_data(input_video.frames[i][...,[2,1,0]])
                points.set_offsets(keypoints[i])

            ###### ///////////////////////////////////////////////////////////
            #ax.scatter(poses[0][i, 1, 0], poses[0][i, 1, 1], 0, c='r', marker='o')  # (255,0,0) poses[0][0, 1, 2]
            #ax.scatter(poses[0][i, 4, 0], poses[0][i, 4, 1], 0, c='r', marker='o')  # poses[0][0, 4, 2]

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
                    lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
                    lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')



        print('{}/{}      '.format(i, limit), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
    if output.endswith('.mp4'):# or output.endswith('.avi'):
        Writer = writers['ffmpeg'] #writers['pillow'] #
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
    print('Video saved')