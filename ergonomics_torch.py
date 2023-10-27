import torch
import matplotlib.pyplot as plt
import numpy as np

class RULATorch():
    def __init__(self, pose_3d):
        self.pose_3d = pose_3d
        self.keypoints = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                          "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
                          "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14,
                          "LAnkle": 15, "RAnkle": 16}

        # table A RULA-Worksheet (without wrist scores)
        self.table_A = [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9]

        # table B RULA-Worksheet
        self.table_B = [[1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
                        [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
                        [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
                        [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
                        [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]]

        # table C RULA-Worksheet
        self.table_C = [[1, 2, 3, 3, 4, 5, 5],
                        [2, 2, 3, 4, 4, 5, 5],
                        [3, 3, 3, 4, 4, 5, 6],
                        [3, 3, 3, 4, 5, 6, 6],
                        [4, 4, 4, 5, 6, 7, 7],
                        [4, 4, 5, 6, 6, 7, 7],
                        [5, 5, 6, 6, 7, 7, 7],
                        [5, 5, 6, 7, 7, 7, 7]]

        self.angle_dict = {0: 'L_shoulder_Z', 1: 'L_shoulder_line', 2: 'R_shoulder_Z',
                           3: 'R_shoulder_line', 4: 'L_elbow', 5: 'R_elbow',
                           6: 'L_knee', 7: 'R_knee', 8: 'Stoop', 9: 'Trunk_twist',
                           10: 'Trunk_sidebend', 11: 'Neck', 12: 'Neck_sidebend',
                           13: 'Neck_twist'}

        self.score_total = torch.tensor([])


    def transform_pose(self, pose):
        """
        takes the original pose and transposes it from XZY to XYZ and flips the Z axes
        :param pose: 3D keypoints
        :return:
        """
        new_pose = pose.permute(1, 0)
        new_pose[[1, 2], :] = new_pose[[2, 1], :]
        new_pose[2, :] = -new_pose[2, :]

        return new_pose

    def accumulate_angles(self):
        all_angles = []

        for ind in range(len(self.pose_3d)):
            angles_frame = []
            pose = self.transform_pose(self.pose_3d[ind].clone()) # Note: clone in PyTorch instead of copy in NumPy

            # left shoulder angles
            shoulder_left_z = self.calculate_z(pose, 11, 12) - 10  #Angle between shoulder, elbow and z-axes (10 degrees
            # substracted as the algorithm leads to around 10 degree exaggeration when standing straight)
            angles_frame.append(shoulder_left_z)
            shoulder_left_shoulderline = self.calculate_angle(pose, 12, 11, 14)  # Angle between elbow and both #90 -
            # shoulders // -10 to 20 deg!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            angles_frame.append(shoulder_left_shoulderline)

            # right shoulder angles
            angle_shoulder_right_z = self.calculate_z(pose, 14, 15) - 10   # Same as left shoulder
            angles_frame.append(angle_shoulder_right_z)
            angle_shoulder_right_shoulderline = self.calculate_angle(pose, 15, 14, 11) #90 -
            angles_frame.append(angle_shoulder_right_shoulderline)


            #/*******************************************************
            #double check...
            # elbow angles
            angle_elbow_left = 180 - self.calculate_angle(pose, 13, 12, 11)  # Winkel zwischen Oberarm und Unterarm
            angles_frame.append(angle_elbow_left)
            angle_elbow_right = 180 - self.calculate_angle(pose, 14, 15, 16)  # Winkel zwischen Oberarm und Unterarm
            angles_frame.append(angle_elbow_right)

            # knee angles
            angle_knee_left = 180 - self.calculate_angle(pose, 4, 5, 6)   # Winkel zwischen Oberschenkel und Unterschenkel
            angles_frame.append(angle_knee_left)
            angle_knee_right = 180 - self.calculate_angle(pose, 1, 2, 3)  # Winkel zwischen Oberschenkel und Unterschenkel
            angles_frame.append(angle_knee_right)

            # trunk
            angle_stoop = 180 - self.calculate_z(pose, 0, 8)   # Neigung der oberen Wirbelsäule zur Senkrechten
            angles_frame.append(angle_stoop)
            angle_trunk_twist = self.calculate_twist(pose, 1, 4, 11, 14)  # Winkel zwischen Schulter- und Hüftachse in der XY-Ebene ///
            angles_frame.append(angle_trunk_twist)
            angle_trunk_sidebending = self.calculate_twist(pose, 1, 4, 0, 7)  # Winkel zwischen Hüftachse und oberer Wirbelsäule # pose, 1, 4, 7, 8
            angles_frame.append(angle_trunk_sidebending)

            # neck
            angle_neck = 180 - self.calculate_z(pose, 8, 10)
            #angle_neck = 180 - self.calculate_angle(pose, 7, 8, 10)  # Winkel zwischen oberer Wirbelsäule und Kopf (Stirn/zentraler Kopf)
            angles_frame.append(angle_neck)
            angle_neck_sidebending = self.calculate_twist(pose, 11, 14, 8, 10)  # Winkel zwischen Schulterachse und Nacken-Nase-Linie in der XY-Ebene
            angles_frame.append(angle_neck_sidebending)
            angle_neck_twist = self.calculate_twist(pose, 11, 14, 9, 10)  # Winkel zwischen Schulterachse und Nacken-Nase-Linie in der XY-Ebene
            angles_frame.append(angle_neck_twist)

            all_angles.append(angles_frame)

        return all_angles

    # ... [rest of your methods]

    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]

        ba = a - b
        bc = c - b

        cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(cosine_angle)

        return angle * (180 / torch.pi)

    def calculate_z(self, pose, joint1, joint2):
        a = pose[:, joint1]
        b = pose[:, joint2]

        ba = a - b
        bc = torch.tensor([0, 0, 1], dtype=torch.float)
        cross_product = torch.cross(ba, bc)
        z_value = cross_product[2]
        return z_value

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]
        d = pose[:, joint4]

        ba = a - b
        dc = d - c

        cosine_angle = torch.dot(ba, dc) / (torch.norm(ba) * torch.norm(dc))
        angle = torch.acos(cosine_angle)
        return angle * (180 / torch.pi)


    def plot_angles(self, angles, title='Joint Angles'):
        """
        Plot joint angles over time.
        :param angles: A PyTorch tensor containing joint angles. Expected shape: [num_frames, num_joints]
        :param title: Title of the plot.
        """

        # Detach tensors and convert to numpy for plotting
        angles_np = angles.detach().cpu().numpy()

        # Assuming angles_np is a 2D array with shape [num_frames, num_joints]
        num_frames, num_joints = angles_np.shape

        plt.figure(figsize=(10, 6))
        for joint_idx in range(num_joints):
            plt.plot(angles_np[:, joint_idx], label=f'Joint {joint_idx + 1}')

        plt.title(title)
        plt.xlabel('Frame')
        plt.ylabel('Angle (in degrees)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_scores(self, fps=None):

        scores = self.score_total

        # If score_total is a tensor, convert it to a numpy array
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        plt.figure(figsize=(16, 6), dpi=120)

        if fps is not None:
            timestamps = np.linspace(0, int(len(scores)/fps), len(scores))
            plt.plot(timestamps, scores, color='b')
        else:
            plt.plot(range(len(scores)), scores, color='b')

        plt.plot(np.array([2]*len(scores)), linestyle='--', color='g')
        plt.plot(np.array([4] * len(scores)), linestyle='--', color='y')
        plt.plot(np.array([6] * len(scores)), linestyle='--', color='r')
        plt.ylim([1,7])

        plt.yticks([1,2,3,4,5,6,7], ['1','2','3','4','5','6','7'], fontsize=20, fontweight='medium')

        if fps is not None:
            plt.xlim([0, timestamps[-1]])
            plt.xticks(fontsize=20, fontweight='medium')
            plt.xlabel('Video duration in seconds', fontsize=25, fontweight='medium')
        else:
            plt.xlim([0, len(scores)])
            plt.xlabel('Video frames')

        plt.ylabel('Ergonomic Risk', fontsize=25, fontweight='medium')

        RULA_txt = ('1-2 = acceptable posture\n3-4 = further investigation, change may be needed\n'
                    '5-6 = further investigation, change soon\n 7 = investigate and implement changes')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.figtext(0.5, 0.2, RULA_txt, fontsize=14, horizontalalignment='center',
                    verticalalignment='top', bbox=props)
        plt.subplots_adjust(bottom=0.35)
        plt.show()

    def compute_scores(self):
        # elementary scores
        score_lower_arm = torch.empty(len(self.accumulate_angles()))
        score_upper_arm = torch.empty(len(self.accumulate_angles()))
        score_trunk = torch.empty(len(self.accumulate_angles()))
        score_neck = torch.empty(len(self.accumulate_angles()))
        score_legs = torch.empty(len(self.accumulate_angles()))

    # global scores
        score_A = torch.empty(len(self.accumulate_angles()))
        score_B = torch.empty(len(self.accumulate_angles()))
        score_total = torch.empty(len(self.accumulate_angles()))

        angles = torch.tensor(self.accumulate_angles())

        # Calculations
        for i in range(angles.size(0)):
            print('i :', i)
            frame = angles[i]

            # A. arms step 1
            max_shoulder = torch.max(torch.abs(frame[0]), torch.abs(frame[2]))
            score_upper_arm[i] = torch.where(max_shoulder <= 20, 1,
                                             torch.where(max_shoulder <= 45, 2,
                                                         torch.where(max_shoulder <= 90, 3, 4)))
            print('Max shoulder, score_upper_arm[i]: ', max_shoulder, ', ', score_upper_arm[i])
            # A. check abduction
            max_abduction = torch.max(frame[1], frame[3])
            score_upper_arm[i] += (max_abduction > 150).float()

            # A. arms step 2
            max_elbow = torch.max(frame[4], frame[5])
            score_lower_arm[i] = torch.where((max_elbow <= 20) | (max_elbow >= 100), 2, 1)

            curr_score_A = self.table_A[(score_upper_arm[i]-1).long()*3 + (score_lower_arm[i] - 1).long()]
            score_A[i] = curr_score_A

            # B. neck
            score_neck[i] = torch.where(frame[11] >= 20, 1,
                                        torch.where(frame[11] >= 40, 2,
                                                    torch.where(frame[11] <= 20, 4, 3)))

            # neck side bending
            score_neck[i] += (frame[12] >= 120 or frame[12] <= 60).float()

            # B. trunk
            score_trunk[i] = torch.where(frame[8] <= 15, 1,
                                         torch.where(frame[8] <= 30, 2,
                                                     torch.where(frame[8] <= 60, 3, 4)))

            # trunk side bending
            score_trunk[i] += (frame[10] <= 60 or frame[10] >= 120).float()
            # trunk twist
            score_trunk[i] += (frame[9] >= 30).float()

            # B. legs
            min_knee = torch.min(frame[6], frame[7])
            score_legs[i] = torch.where(min_knee >= 90, 2, 1)

            curr_score_B = self.table_B[(score_neck[i]-1).long()][(score_trunk[i]-1).long()*2 + (score_legs[i]-1).long()]
            score_B[i] = curr_score_B

            curr_score_A = torch.minimum(torch.tensor(8.0), torch.tensor(curr_score_A, dtype=torch.float))
            curr_score_B = torch.minimum(torch.tensor(7.0), torch.tensor(curr_score_B, dtype=torch.float))
            print("Current score A B : ", curr_score_A, curr_score_B)
            curr_total = self.table_C[(curr_score_A-1).long()][(curr_score_B-1).long()]
            score_total[i] = curr_total

        self.score_total = score_total

        return score_total

