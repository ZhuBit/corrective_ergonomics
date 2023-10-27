from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from OC_SORT.ocsort import OCSort

import numpy as np
import time
import math

#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#import matplotlib
#print(matplotlib.get_backend())

class Wrapper_2Dpose():

    def __init__(self, model, weights, ROI_thr, cuda=True, tracker='sort'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = ROI_thr
        self.thr = ROI_thr
        self.cfg.MODEL.WEIGHTS = weights
        self.cfg.INPUT.MIN_SIZE_TEST = 400
        if cuda == False:
            self.cfg.MODEL.DEVICE = "cpu"
        #self.predictor = DefaultPredictor(self.cfg)
        self.tracker = tracker

    def calculate_IoU(self, BB1, BB2):

        xA = max(BB1[0], BB2[0])
        yA = max(BB1[1], BB2[1])
        xB = min(BB1[2], BB2[2])
        yB = min(BB1[3], BB2[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0

        boxAArea = abs((BB1[2] - BB2[0]) * (BB1[3] - BB2[1]))
        boxBArea = abs((BB2[2] - BB2[0]) * (BB2[3] - BB2[1]))

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def simple_tracker(self, out_t1, out_t0):

        max_iou = math.inf
        person = 0
        for i in range(out_t1.pred_boxes.tensor.numpy().shape[0]):
            iou = self.calculate_IoU(BB1=out_t1.pred_boxes.tensor.numpy()[i], BB2=out_t0) #videtensor
            if iou > max_iou:
                max_iou = iou
                person = i

        return person

    def predict_2D_poses(self, input_video_object, output_dir='/content'):

        boxes = []

        keypoints = []

        self.cfg.INPUT.MIN_SIZE_TEST = min(input_video_object.h, input_video_object.w)
        self.predictor = DefaultPredictor(self.cfg)


        detected = False #whether a person was detected in a video
        person_index = 0 #index for tracking

        if self.tracker == 'sort':
            tracker = OCSort(det_thresh=0.6, iou_threshold=0.2, use_byte=False, max_age=input_video_object.fps*3,
                             delta_t=input_video_object.fps, min_hits=5)
            tracker_size = [input_video_object.h, input_video_object.w]
            tracked_id = None

        for frame_i, im in enumerate(input_video_object.frames):
            t = time.time()

            outputs = self.predictor(im)['instances'].to('cpu')


            has_bbox = False
            if detected == False: #initial detection - no person detected before
                if len(outputs.scores) > 1: #more than 1 person in initial detection
                    plt.imshow(im[...,[2,1,0]]) #np.flip(
                    ax = plt.gca()
                    for p in range(len(outputs.scores)):
                        bb_p = outputs.pred_boxes[p].tensor.numpy().squeeze(0)
                        rec = Rectangle((int(bb_p[0]), int(bb_p[1])), int(bb_p[2]-bb_p[0]), int(bb_p[3]-bb_p[1]),
                                         linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rec)
                        scale_w, scale_h = [int(input_video_object.h*0.05), int(input_video_object.h*0.05)]
                        plt.text(int(bb_p[0])-scale_w, int(bb_p[3])+scale_h, str(p), color='r')
                    plt.text(int(input_video_object.h*0.1), int(input_video_object.w*0.1),
                             '{} detected workers'.format(len(outputs.scores)), color='r')
                    plt.draw()
                    plt.pause(0.001)
                    input('Press Enter to continue')
                    #plt.show()

                    person_index = len(outputs.scores)
                    while person_index < 0 or person_index > len(outputs.scores)-1:
                        person_index = int(input('Input the ID of the worker to track - '))

                    bbox_tensor = outputs.pred_boxes[person_index].tensor.squeeze(0).numpy()
                    scores = outputs.scores.numpy()[person_index, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=0)

                    if self.tracker == 'sort':
                        output_results = np.concatenate((outputs.pred_boxes.tensor.numpy(), np.expand_dims(outputs.scores.numpy(), axis=1)), axis=1)
                        tracked = tracker.update(output_results, tracker_size, tracker_size)
                    else:
                        previous_outputs = outputs.pred_boxes.tensor.numpy()[person_index]  # for tracking
                    detected = True
                    has_bbox = True
                elif len(outputs.scores) == 1: #one person in the first frame
                    bbox_tensor = outputs.pred_boxes[person_index].tensor.squeeze(0).numpy()
                    scores = outputs.scores.numpy()[person_index, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=0)


                    person_index = np.argmax(scores) #outputs.scores
                    if self.tracker == 'sort':
                        output_results = np.concatenate(
                            (outputs.pred_boxes.tensor.numpy(), np.expand_dims(outputs.scores.numpy(), axis=1)), axis=1)
                        tracked = tracker.update(output_results, tracker_size, tracker_size)
                    else:
                        previous_outputs = outputs.pred_boxes.tensor.numpy()[person_index]  # for tracking
                    detected = True
                    has_bbox = True
                else: #no person detected
                    boxes.append(np.empty((0,5)))
                    keypoints.append(np.empty((17,3)))


            else: #a person was previously detected
                if len(outputs.scores) > 0: #detection in current frame
                    if self.tracker == 'sort':
                        output_results = np.concatenate((outputs.pred_boxes.tensor.numpy(),
                                                         np.expand_dims(outputs.scores.numpy(), axis=1)), axis=1)
                        tracked = tracker.update(output_results, tracker_size, tracker_size)
                        tracked_detection = tracked[tracked[:, 4] == person_index + 1]


                        if len(tracked_detection) > 0:
                            if len(outputs) == 1:
                                bbox_tensor = outputs.pred_boxes.tensor.squeeze(0).numpy()
                            else:
                                bbox_tensor = outputs.pred_boxes.tensor.squeeze(0).numpy()[person_index]

                            scores = outputs.scores.numpy()[person_index, None]
                            bbox_tensor = np.concatenate((bbox_tensor, scores), axis=0)
                            has_bbox = True

                    else:  # IoU tracker
                        person_index = self.simple_tracker(out_t1=outputs, out_t0=previous_outputs)
                        previous_outputs = outputs.pred_boxes.tensor.numpy()[person_index]  # for tracking

                        bbox_tensor = outputs.pred_boxes[person_index].tensor.numpy()  # [person_index].unsqueeze(0)
                        #bbox_tensor = np.expand_dims(bbox_tensor[person_index], 0)
                        if len(bbox_tensor) > 0:
                            has_bbox = True
                            scores = outputs.scores.numpy()[:, None]
                            bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
                else: #no detection in current frame
                    pass


            if has_bbox: #
                if self.tracker == 'sort' and tracked_id is not None:
                    kps = outputs.pred_keypoints[tracked_id].unsqueeze(0).numpy()
                else:
                    kps = outputs.pred_keypoints[person_index].unsqueeze(0).numpy()

                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]

                kps = np.concatenate((kps_xy, kps_prob), axis=2)
                kps = kps.squeeze(0)

            else:
                kps = np.zeros((17,3))
                bbox_tensor = np.zeros((5))


            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

            boxes.append(bbox_tensor)
            keypoints.append(kps)

        # Video resolution TODO only works with coco dataset, this is metatada
        # if we dont use 2d psoe esti, we need to reconctruct, this
        resolution = {
            'w': im.shape[1],
            'h': im.shape[0],
            'fps': input_video_object.fps
        }
        metadata = {}
        metadata['layout_name'] = 'coco'
        metadata['num_joints'] = 17
        metadata['keypoints_symmetry'] = [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]
        metadata['video_metadata'] = {'data_2d': resolution}

        bb = np.array(boxes, dtype=np.float32)
        bb_check = bb.copy()
        bb_check[bb_check[:,0] == 0] = 'nan'
        mask = ~np.isnan(bb_check[:, 0])  # .flatten()
        kp = np.array(keypoints, dtype=np.float32)

        indices = np.arange(len(bb))
        #print(indices.shape)
        for i in range(4):
            bb[:, i] = np.interp(indices, indices[mask], bb[mask,i])
        for i in range(17):
            for j in range(2):
                kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])

        keypoints = {'data_2d':{'custom':kp}}

        print("Finished processing video!")
        # NEEED TO PACK THIS it can be done with pack_3D_keypoints()
        return {
            'start_frame': 0,  # Inclusive
            'end_frame': len(keypoints),  # Exclusive
            'bounding_boxes': bb,  # boxes
            'keypoints': keypoints,  # keypoints
        }, metadata

from common.mocap_dataset import MocapDataset
from common.h36m_dataset import h36m_skeleton
from common.camera import normalize_screen_coordinates

custom_camera_params = {
    'id': None,
    'res_w': None,  # Pulled from metadata
    'res_h': None,  # Pulled from metadata

    # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
    'azimuth': 70,  # Only used for visualization
    'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
    'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
}


class Dataset_Wrapper(MocapDataset):
    """
        Adapted based on the work of Dario Pavllo
        https://github.com/facebookresearch/VideoPose3D
    """
    def __init__(self, data, metadata, remove_static_joints=True):
        super().__init__(fps=None, skeleton=h36m_skeleton)

        # Load serialized dataset
        resolutions = metadata['video_metadata']

        self._cameras = {}
        self._data = {}
        import copy
        self._poses = copy.deepcopy(data) #data
        for video_name, res in resolutions.items():
            cam = {}
            cam.update(custom_camera_params)
            cam['orientation'] = np.array(cam['orientation'], dtype='float32')
            cam['translation'] = np.array(cam['translation'], dtype='float32')
            cam['translation'] = cam['translation'] / 1000  # mm to meters

            cam['id'] = video_name
            cam['res_w'] = res['w']
            cam['res_h'] = res['h']

            self._cameras[video_name] = [cam]

            self._data[video_name] = {
                'custom': {
                    'cameras': cam
                }
            }

        if remove_static_joints and len(self._skeleton._parents) > 17:
            # Bring the skeleton to 17 joints instead of the original 32
            self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])

            # Rewire shoulders to the correct parents
            self._skeleton._parents[11] = 8
            self._skeleton._parents[14] = 8

    def supports_semi_supervised(self):
        return False

    def fetch(self):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []

        poses_2d = self._poses['keypoints']['data_2d']['custom'][...,:2]
        cam = self._cameras['data_2d'][0]
        poses_2d[..., :2] = normalize_screen_coordinates(poses_2d[..., :2], w=cam['res_w'], h=cam['res_h']) #self._cameras


        out_poses_2d.append(poses_2d)

        if 'intrinsic' in self._cameras:
            out_camera_params.append(self._cameras['intrinsic'])

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None


        return out_camera_params, out_poses_3d, out_poses_2d


import torch
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator


class Wrapper_3Dpose():
    """
        Adapted based on the work from Dario Pavllo
        https://github.com/facebookresearch/VideoPose3D
    """

    def __init__(self, chck_path): #, data, metadata, render=False, input_vid='video.mp4', output_vid='output_video.mp4'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        chk_filename = chck_path
        #print('Loading checkpoint')

        self.checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        #print('This model was trained for {} epochs'.format(self.checkpoint['epoch']))

    def _pose_transformation(self, pose):
        def calc_0_translation(pose):
            x_trans = pose[0, 0] - 0
            y_trans = pose[0, 1] - 0
            translation = [[x_trans, y_trans, 0] for i in range(pose.shape[0])]
            return translation

        def calc_rotation(pose):
            point_L = np.asarray([pose[4, 0], pose[4, 1], 0])
            point_R = np.asarray([pose[1, 0], pose[1, 1], 0])

            angle = math.atan2((point_R[1]-point_L[1]), (point_R[0]-point_L[0]))
            return angle

        def rotation_matrix(angle):
            axis = [0, 0, 1]
            axis = np.asarray(axis)
            # axis = axis / math.sqrt(np.dot(axis, axis))
            a = math.cos(angle / 2.0)
            b, c, d = axis * math.sin(angle / 2.0)
            aa, bb, cc, dd = a * a, b * b, c * c, d * d
            bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
            return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                             [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                             [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

        def rotate(X, angle):
            '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
            c, s = np.cos(angle), np.sin(angle)
            return np.dot(X, np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1.],
            ]))


        pose = np.asarray(pose)
        trans = calc_0_translation(pose)
        new_pose = pose - trans
        angle = calc_rotation(new_pose)

        new_pose = rotate(new_pose, angle)
        new_pose = rotate(new_pose, math.pi/2) #project parallel to y --> better visibility

        new_pose += trans
        return new_pose

    def render_video_output(self, output_path, video_object=None, transform_coronal=False):
        if self.results is None:
            raise ValueError("3D data has to be first generated to render a video")

        cam = self.dataset.cameras()['data_2d'][0]

        input_keypoints = image_coordinates(self.poses_valid_2d[0][:, :, :2], w=cam['res_w'], h=cam['res_h'])

        rot = cam['orientation']
        prediction = camera_to_world(self.results, R=rot, t=0) #.squeeze(0).cpu().numpy()
        prediction[:, :, 2] -= np.min(prediction[:, :, 2])

        transformed = prediction.copy()

        if transform_coronal:
            transformed = np.asarray([self._pose_transformation(prediction[i,...]) for i in range(prediction.shape[0])])

        anim_output = {'3D Reconstruction': transformed} #

        from visualiser import render_animation_custom
        render_animation_custom(input_keypoints, self.metadata, anim_output,
                              self.dataset.skeleton(), self.dataset.fps(), 3000, 70, output_path,  # cam['azimuth']
                              limit=-1, size=5, input_video=video_object, viewport=(cam['res_w'], cam['res_w']))

    def predict_3D_poses(self, data, metadata): # pack_3D_keypoints, +metadata todo

        print("Lifting 2D Poses into 3D...")
        self.metadata = metadata
        self.data = data
        self.dataset = Dataset_Wrapper(data, metadata)

        cameras_valid, poses_valid, poses_valid_2d = self.dataset.fetch()
        self.poses_valid_2d = poses_valid_2d

        model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 17,
                                  filter_widths=[3, 3, 3, 3, 3])
        model_pos.load_state_dict(self.checkpoint['model_pos'])

        model_pos.to(self.device)
        model_pos.eval()

        traj = False
        if 'model_traj' in self.checkpoint:
            model_traj = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                                       filter_widths=[3, 3, 3, 3, 3])
            model_traj.load_state_dict(self.checkpoint['model_traj'])
            model_traj.to(self.device)
            #print('loaded trajectory model')
            traj = True

        receptive_field = model_pos.receptive_field()
        #print('INFO: Receptive field: {} frames'.format(receptive_field))
        pad = (receptive_field - 1) // 2
        causal_shift = 0

        keypoints_symmetry = self.metadata['keypoints_symmetry']
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        joints_left, joints_right = list(self.dataset.skeleton().joints_left()), list(self.dataset.skeleton().joints_right())

        cameras = []
        poses_valid_3d = []

        test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                            pad=pad, causal_shift=causal_shift, augment=False,
                                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                            joints_right=joints_right)


        with torch.no_grad():
            if not traj:
                model_pos.eval()
            else:
                model_traj.eval()
            N = 0

            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

                # Positional model
                if not traj:
                    predicted_3d_pos = model_pos(inputs_2d.to(self.device))
                else:
                    predicted_3d_pos = model_traj(inputs_2d.to(self.device))

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    if not use_trajectory_model:
                        predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :,
                                                                             joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

                results = predicted_3d_pos  # .squeeze(0).cpu().numpy()

                self.results = results.squeeze().cpu().numpy()

                print("Finished 3D pose reconstruction!")
                # RODO OUTPUTS 3D PART WOHTOUT 3D COORDINATE COORDINATE
                return results.squeeze().cpu().numpy()
                # UCNOMENT THIS TO HAEV TRANFORMED 3D COORDINATES TODO
                # CREATE OWN VIZUALZITION
                ##///////////////////////////////////

                # res = results.squeeze().cpu().numpy().copy()
                # cam = self.dataset.cameras()['data_2d'][0]
                #
                # #input_keypoints = image_coordinates(poses_valid_2d[0][:, :, :2], w=cam['res_w'], h=cam['res_h'])
                #
                # rot = cam['orientation']
                # prediction = camera_to_world(results.squeeze(0).cpu().numpy(), R=rot, t=0)
                # prediction[:, :, 2] -= np.min(prediction[:, :, 2])
                #
                # # transformed = prediction.copy()
                #
                # # prediction = np.asarray(self._pose_transformation(prediction[i, ...]))
                # transformed = np.asarray(
                        #POSE TANFOR IS NOT OPTMAL TODO
                #     [self._pose_transformation(prediction[i, ...]) for i in range(prediction.shape[0])])
                # return [transformed, res]



