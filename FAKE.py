import torch
import torch.nn as nn
import torch.nn.functional as F

class FAKE(nn.Module):
    def __init__(self, device):
        super(FAKE, self).__init__()
        #self.target = target.detach()  # Detach the target as it's a fixed value
        self.device = device
        self.keypoints = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                          "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
                          "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14,
                          "LAnkle": 15, "RAnkle": 16}

        # table A RULA-Worksheet (without wrist scores)
        self.table_A = torch.tensor([1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9],
                                    dtype=torch.float32, requires_grad=True, device=self.device)

        # table B RULA-Worksheet
        self.table_B = torch.tensor([[1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
                                     [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
                                     [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
                                     [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
                                     [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
                                     [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]],
                                    dtype=torch.float32, requires_grad=True, device=self.device)

        # table C RULA-Worksheet
        self.table_C = torch.tensor([[1, 2, 3, 3, 4, 5, 5],
                                     [2, 2, 3, 4, 4, 5, 5],
                                     [3, 3, 3, 4, 4, 5, 6],
                                     [3, 3, 3, 4, 5, 6, 6],
                                     [4, 4, 4, 5, 6, 7, 7],
                                     [4, 4, 5, 6, 6, 7, 7],
                                     [5, 5, 6, 6, 7, 7, 7],
                                     [5, 5, 6, 7, 7, 7, 7]],
                                    dtype=torch.float32, requires_grad=True, device=self.device)

        self.angle_dict = {0: 'L_shoulder_Z', 1: 'L_shoulder_line', 2: 'R_shoulder_Z',
                           3: 'R_shoulder_line', 4: 'L_elbow', 5: 'R_elbow',
                           6: 'L_knee', 7: 'R_knee', 8: 'Stoop', 9: 'Trunk_twist',
                           10: 'Trunk_sidebend', 11: 'Neck', 12: 'Neck_sidebend',
                           13: 'Neck_twist'}

        self.score_total = torch.tensor([], requires_grad=True, dtype=torch.float32, device=self.device)

    def forward(self, pose_3d):
        # Compute RULA scores from pose_3d
        #pose_3d = pose_3d.to(self.device)
        total_score = self.compute_scores(pose_3d)

        # Compute the loss as the mean squared error between
        # the computed scores and the target scores
        #loss = F.mse_loss(total_score, pose_3d, reduction='sum')

        return torch.sum(total_score, dim=0) #loss
    def calculate_neck_score(self, neck_angle, steepness):
        """
        :param frame: angle of the neck for one frame
        :return: score for the neck
        """
        if not isinstance(neck_angle, torch.Tensor):
            neck_angle = torch.tensor(neck_angle, dtype=torch.float32, requires_grad=True)
        # Parameters to adjust the steepness and position of the sigmoid transitions
        # Transition for the condition frame >= 40
        score_for_40 = 2 * torch.sigmoid(steepness * (neck_angle - 40))

        # Transition for the condition frame >= 20 and frame < 40
        score_for_20 = 1 * torch.sigmoid(steepness * (neck_angle - 20))

        # Transition for the condition frame < 20
        score_for_less_20 = 4 - 4 * torch.sigmoid(steepness * (neck_angle - 20))

        # Combining the scores
        score = torch.max(torch.max(score_for_40, score_for_20), score_for_less_20)

        return score

    def calculate_side_bending_scores(self, side_angle, steepness):
        """
        Calculate RULA scores for neck side bending based on a series of frames using a sigmoid function.
        :param frames: A list or array of neck angles for different frames.
        :return: A list of scores for each frame.
        """
        if not isinstance(side_angle, torch.Tensor):
            side_angle = torch.tensor(side_angle, dtype=torch.float32, requires_grad=True)

        # Parameters for sigmoid function got from RULA sheet
        lower_bound = 60
        upper_bound = 120

        # Apply sigmoid function to approximate the score
        scores = 1 - torch.sigmoid(steepness * (side_angle - lower_bound)) * torch.sigmoid(steepness * (-side_angle + upper_bound))

        return scores

    def calculate_trunk_scores(self, trunk_angle, steepness):
        """
        Calculate RULA scores for trunk bending based on a series of frames using sigmoid functions.
        :param frames: A list or array of trunk angles for different frames.
        :return: A list of scores for each frame.
            """
        if not isinstance(trunk_angle, torch.Tensor):
            frames = torch.tensor(trunk_angle, dtype=torch.float32, requires_grad=True)

        # Defining the thresholds for different RULA scores
        thresholds = [15, 30, 60]

        # Applying sigmoid functions to approximate the score
        score_for_15 = 1 + 1 * torch.sigmoid(steepness * (trunk_angle - thresholds[0]))
        score_for_30 = 2 + 1 * torch.sigmoid(steepness * (trunk_angle - thresholds[1]))
        score_for_60 = 3 + 1 * torch.sigmoid(steepness * (trunk_angle - thresholds[2]))

        # Selecting the highest score that still meets the condition
        scores = torch.min(torch.min(score_for_15, score_for_30), score_for_60)

        return scores
    def trunk_twist(self, trunk_angle, steepness):
        """
        Calculate RULA scores for trunk twisting based on a series of angles using a sigmoid function.
        """
        if not isinstance(trunk_angle, torch.Tensor):
            trunk_angle = torch.tensor(trunk_angle, dtype=torch.float32, requires_grad=True)
        # Threshold for triggering an increase in score
        threshold = 30
        # Applying a sigmoid function to approximate the score increase
        scores = torch.sigmoid(steepness * (trunk_angle - threshold))
        return scores


    def compute_scores(self, pose_3d):

        score_lower_arm = torch.tensor([], requires_grad=True, dtype=torch.float32, device=self.device)
        score_trunk = torch.tensor([], requires_grad=True, dtype=torch.float32, device=self.device)
        score_neck = torch.tensor([], requires_grad=True, dtype=torch.float32, device=self.device)
        score_legs = torch.tensor([], requires_grad=True, dtype=torch.float32, device=self.device)

        score_upper_arm = torch.empty((0, 1), requires_grad=True, dtype=torch.float32, device=self.device)

        score_total = []
        simple_angel_score = []
        frames_angles = self.accumulate_frames_angles(pose_3d)

        steepness = 5
        for i, f_angels in enumerate(frames_angles):
            print('--------------Frame:', i)
            # Neck
            score_neck = self.calculate_neck_score(f_angels[11], steepness)
            score_neck += self.calculate_side_bending_scores(f_angels[12], steepness)
            # Trunk
            score_trunk = self.calculate_trunk_scores(f_angels[8], steepness)
            score_trunk += self.calculate_side_bending_scores(f_angels[10], steepness)
            score_trunk += self.trunk_twist(f_angels[9], steepness)
            print('Trunk Angle: ', f_angels[8].item())
            print('Trunk: ', score_trunk.item())

            simple_angel_score.append(score_neck)
            
        simple_angel_score = torch.stack(simple_angel_score, dim=0)
        return simple_angel_score

    def accumulate_frames_angles(self, pose_3d):

        """
        computes the angles between body parts as specified by the RULA worksheet
        :return: angles between body parts
        """
        all_angles = []

        for ind in range(len(pose_3d)):
            angles_frame = []

            pose = self.transform_pose(pose_3d[ind,...].clone()) # deep copy

            # left shoulder angles
            shoulder_left_z = self.calculate_z(pose, 11, 12) - 10
            angles_frame.append(shoulder_left_z)
            shoulder_left_shoulderline = self.calculate_angle(pose, 12, 11, 14)
            angles_frame.append(shoulder_left_shoulderline)

            # right shoulder angles
            angle_shoulder_right_z = self.calculate_z(pose, 14, 15) - 10
            angles_frame.append(angle_shoulder_right_z)
            angle_shoulder_right_shoulderline = self.calculate_angle(pose, 15, 14, 11)
            angles_frame.append(angle_shoulder_right_shoulderline)

            # elbow angles
            angle_elbow_left = 180 - self.calculate_angle(pose, 13, 12, 11)
            angles_frame.append(angle_elbow_left)
            angle_elbow_right = 180 - self.calculate_angle(pose, 14, 15, 16)
            angles_frame.append(angle_elbow_right)

            # knee angles
            angle_knee_left = 180 - self.calculate_angle(pose, 4, 5, 6)
            angles_frame.append(angle_knee_left)
            angle_knee_right = 180 - self.calculate_angle(pose, 1, 2, 3)
            angles_frame.append(angle_knee_right)

            # trunk
            angle_stoop = 180 - self.calculate_z(pose, 0, 8)
            angles_frame.append(angle_stoop)
            angle_trunk_twist = self.calculate_twist(pose, 1, 4, 11, 14)
            angles_frame.append(angle_trunk_twist)
            angle_trunk_sidebending = self.calculate_twist(pose, 1, 4, 0, 7)
            angles_frame.append(angle_trunk_sidebending)

            # neck
            angle_neck = 180 - self.calculate_z(pose, 8, 10)
            angles_frame.append(angle_neck)
            angle_neck_sidebending = self.calculate_twist(pose, 11, 14, 8, 10)
            angles_frame.append(angle_neck_sidebending)
            angle_neck_twist = self.calculate_twist(pose, 11, 14, 9, 10)
            angles_frame.append(angle_neck_twist)

            #angles_frame_tensor = torch.tensor(angles_frame, requires_grad=True)
            angles_frame_tensor = torch.stack(angles_frame)
            all_angles.append(angles_frame_tensor)

        return torch.stack(all_angles, dim=0)

    def transform_pose(self, pose):
        """
        takes the original pose and transposes it from XZY to XYZ and flips the Z axes
        :param pose: 3D keypoints
        :return:
        """
        # Use torch's transpose method instead of np.transpose
        new_pose = torch.transpose(pose, 0, 1)

        # Swap the second and third dimensions
        #new_pose[1, :], new_pose[2, :] = new_pose[2, :].clone(), new_pose[1, :].clone()
        temp = new_pose[1, :].clone()
        new_pose[1, :] = new_pose[2, :].clone()
        new_pose[2, :] = temp

        # Flip the Z axis
        #new_pose[2, :] = -new_pose[2, :]
        new_pose[2, :] = new_pose[2, :].neg()

        return new_pose

    def calculate_z(self, pose, joint1, joint2):
        a = pose[:, joint1]
        b = pose[:, joint2]

        ba = a - b
        bc = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        """cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(cosine_angle)"""
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        angle_degrees = angle * (180 / torch.pi)

        return angle_degrees

    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]

        ba = a - b
        bc = c - b

        # Ensure ba and bc are 1-dimensional or use torch.matmul for multi-dimensional tensors
        cosine_angle = torch.sum(ba * bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]
        d = pose[:, joint4]

        ba = a - b
        dc = d - c

        # Use torch.sum for dot product to handle tensors of any shape
        cosine_angle = torch.sum(ba * dc) / (torch.norm(ba) * torch.norm(dc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)

