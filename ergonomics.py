import math
import numpy as np
#import matplotlib

#print(matplotlib.get_backend())
#matplotlib.use('Qt5Agg') #GTK3Agg
#matplotlib.use("GTK3Agg")
#matplotlib.use('TkAgg')


import matplotlib.pyplot as plt

class RULA():
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

        self.score_total = list()

    def transform_pose(self, pose):
        """
        takes the original pose and transposes it from XZY to XYZ and flips the Z axes
        :param pose: 3D keypoints
        :return:
        """
        #print('3 pose.shape: ', pose.shape)
        new_pose = np.transpose(pose, (1, 0))
        new_pose[[1, 2], :] = new_pose[[2, 1], :]
        new_pose[2, :] = -new_pose[2, :]

        return np.asarray(new_pose)


    def accumulate_angles(self):
        """
        computes the angles between body parts as specified by the RULA worksheet
        :return: angles between body parts
        """
        all_angles = []

        for ind in range(len(self.pose_3d)):
            angles_frame = []
            #print('0. self.pose_3d type: ', type(self.pose_3d.shape))
            #rint('1. self.pose_3d.shape: ', self.pose_3d.shape)
            #print('2. self.pose_3d[ind,...].shape: ', self.pose_3d[ind,...].shape)
            pose = self.transform_pose(self.pose_3d[ind,...]) # deep copz

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

    # 0 Hüftmitte
    # 1 rechte Hüfte
    # 2 rechtes Knie
    # 3 rechter Fuß
    # 4 linke Hüfte
    # 5 linkes Knie
    # 6 linker Fuß
    # 7 Wirbelsäule
    # 8 Nacken
    # 9 Nase
    # 10 Stirn
    # 11 linke Schulter
    # 12 linker Ellbogen
    # 13 linke Hand
    # 14 rechte Schulter
    # 15 rechter Ellbogen
    # 16 rechte Hand

    def compute_scores(self):
        # Elementary scores
        score_lower_arm = list()
        score_upper_arm = list()
        score_trunk = list()
        score_neck = list()
        score_legs = list()

        # Global scores
        score_A = list()
        score_B = list()
        score_total = list()

        angles = self.accumulate_angles()
        #print('Angles:', angles)

        for i, frame in enumerate(angles):
            print('--------------Frame:', i)

            # A. Arms Step 1
            max_shoulder = max(abs(frame[0]), abs(frame[2]))
            #print('Max shoulder:', max_shoulder)

            if max_shoulder <= 20:
                score_upper_arm.append(1)
            elif max_shoulder <= 45:
                score_upper_arm.append(2)
            elif max_shoulder <= 90:
                score_upper_arm.append(3)
            else:
                score_upper_arm.append(4)

            # A. Check Abduction
            max_abduction = max(frame[1], frame[3])
            #print('Max abduction:', max_abduction)

            if max_abduction > 150:
                score_upper_arm[i] += 1

            # A. Arms Step 2
            max_elbow = max(frame[4], frame[5])
            #print('Max elbow:', max_elbow)

            if max_elbow <= 20 or max_elbow >= 100:
                score_lower_arm.append(2)
            else:
                score_lower_arm.append(1)

            #print('Score Upper Arm:', score_upper_arm)
            #print('Score Lower Arm:', score_lower_arm)

            curr_score_A = self.table_A[(score_upper_arm[i]-1)*3 + (score_lower_arm[i] - 1)]
            #print('Current Score A:', curr_score_A)
            score_A.append(curr_score_A)

            # B. Neck
            if frame[11] >= 20:
                score_neck.append(1)
            elif frame[11] >= 40:
                score_neck.append(2)
            elif frame[11] <= 20:
                score_neck.append(4)
            else:
                score_neck.append(3)

            # Neck Side Bending
            if frame[12] >= 120 or frame[12] <= 60:
                score_neck[i] += 1
            #print('Score Neck:', score_neck)

            # B. Trunk
            print('Trunk Angle: ', frame[8])
            if frame[8] <= 15:
                score_trunk.append(1)
                print('Score Trunk +1')
            elif frame[8] <= 30:
                score_trunk.append(2)
                print('Score Trunk +2')
            elif frame[8] <= 60:
                score_trunk.append(3)
                print('Score Trunk +3')
            else:
                score_trunk.append(4)
                print('Score Trunk +4')

            # Trunk Side Bending
            if frame[10] <= 60 or frame[10] >= 120:
                score_trunk[i] += 1
                print('++++++++++++++++++++Trunk Bend +1')
            # Trunk Twist
            if frame[9] >= 30:
                score_trunk[i] += 1
                print('+++++++++++++++++++++++++++++Trunk Twist +1')

            #print('Score Trunk:', score_trunk)

            # B. Legs
            min_knee = min(frame[6], frame[7])
            #print('Min knee:', min_knee)

            if min_knee >= 90:
                score_legs.append(2)
            else:
                score_legs.append(1)
            #print('Score Legs:', score_legs)

            curr_score_B = self.table_B[(score_neck[i] - 1)][(score_trunk[i] - 1)*2 + (score_legs[i] - 1)]
            #print('Current Score B:', curr_score_B)
            score_B.append(curr_score_B)

            curr_score_A = min(8, curr_score_A)
            curr_score_B = min(7, curr_score_B)

            curr_total = self.table_C[(curr_score_A - 1)][(curr_score_B - 1)]
            #print('Current Total Score:', curr_total)
            score_total.append(curr_total)
            #print('Total Score:', score_total)
        self.score_total = score_total
        return score_total




        # twist --> if sidebending also twist, therefore twist only if no sidebending!!!


    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = np.asarray([pose[0, joint1], pose[1, joint1], pose[2, joint1]])
        b = np.asarray([pose[0, joint2], pose[1, joint2], pose[2, joint2]])
        c = np.asarray([pose[0, joint3], pose[1, joint3], pose[2, joint3]])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        # if cosine_angle < 0:
        #     print(cosine_angle)

        angle = np.arccos(cosine_angle)
        return math.degrees(angle)

    def calculate_z(self, pose, joint1, joint2):
        a = np.asarray([pose[0, joint1], pose[1, joint1], pose[2, joint1]])
        b = np.asarray([pose[0, joint2], pose[1, joint2], pose[2, joint2]])

        ba = a - b
        bc = [0,0,1]

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        # angle = np.math.atan2(np.linalg.det([ba, bc]), np.dot(ba, bc))
        return math.degrees(angle)

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        a = np.asarray([pose[0, joint1], pose[1, joint1], pose[2, joint1]])
        b = np.asarray([pose[0, joint2], pose[1, joint2], pose[2, joint2]])
        c = np.asarray([pose[0, joint3], pose[1, joint3], pose[2, joint3]])
        d = np.asarray([pose[0, joint4], pose[1, joint4], pose[2, joint4]])

        ba = a - b
        dc = d - c

        cosine_angle = np.dot(ba, dc) / (np.linalg.norm(ba) * np.linalg.norm(dc))
        angle = np.arccos(cosine_angle)
        return math.degrees(angle)

    def plot_angles(self, angles, index):
        #matplotlib.use('TkAgg')
        data = np.asarray(angles)[:,index]
        plt.plot(range(len(angles)), np.asarray(data))
        plt.title('Angles for - ' + str(self.angle_dict[index]))
        plt.show()

    def plot_scores(self, fps=None):

        #matplotlib.use('TkAgg')

        # # paper viz
        # self.score_total = self.score_total[:1500]
        # fps = 3

        plt.figure(figsize=(16, 6), dpi=120) #paper viz (16,12)
        if not fps is None:
            timestamps = np.linspace(0, int(len(self.score_total)/fps), len(self.score_total))
            plt.plot(timestamps, self.score_total, color='b')
        else:
            plt.plot(range(len(self.score_total)), self.score_total, color='b') #legend='ergonomic score',
        plt.plot(np.array([2]*len(self.score_total)), linestyle='--', color='g')
        plt.plot(np.array([4] * len(self.score_total)), linestyle='--', color='y')
        plt.plot(np.array([6] * len(self.score_total)), linestyle='--', color='r')
        #plt.plot(range(len(self.score_total)), self.score_total)
        #plt.title('Overall ergonomic score over all video frames')
        plt.ylim([1,7])
        # paper viz
        plt.yticks([1,2,3,4,5,6,7], ['1','2','3','4','5','6','7'], fontsize=20, fontweight='medium')#[1,2,3,5] '3\n (medium\n risk)', '5\n (high\n risk)'])
        if not fps is None:
            plt.xlim([0, timestamps[-1]])

            # paper viz
            plt.xticks(fontsize=20, fontweight='medium')

            plt.xlabel('Video duration in seconds', fontsize=25, fontweight='medium') #paper viz
        else:
            plt.xlim([0, len(self.score_total)])
            plt.xlabel('Video frames')
        plt.ylabel('Ergonomic Risk', fontsize=25, fontweight='medium')

        RULA_txt = '1-2 = acceptable posture\n3-4 = further investigation, change may be needed\n' \
                   '5-6 = further inverstigation, change soon\n 7 = investigate and implement changes'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #plt.gcf().text(0.99, 0.01, RULA_txt, fontsize=14, verticalalignment='top', bbox=props) #0.05, 0.95,
        plt.figtext(0.5, 0.2, RULA_txt, fontsize=14, horizontalalignment='center',
                   verticalalignment='top', bbox=props)
        plt.subplots_adjust(bottom=0.35)
        plt.show()# block=True)



# if __name__ == '__main__':
#     pass
#     #