import torch
import matplotlib.pyplot as plt
import numpy as np

class RULAXXX():
    def __init__(self, pose_3d):

        self.pose_3d = torch.tensor(pose_3d, requires_grad=True, dtype=torch.float32)

        self.keypoints = {"Nose": 0, "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
                          "LShoulder": 5, "RShoulder": 6, "LElbow": 7, "RElbow": 8, "LWrist": 9,
                          "RWrist": 10, "LHip": 11, "RHip": 12, "LKnee": 13, "RKnee": 14,
                          "LAnkle": 15, "RAnkle": 16}

        # table A RULA-Worksheet (without wrist scores)
        self.table_A = torch.tensor([1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9], dtype=torch.float32)

        # table B RULA-Worksheet
        self.table_B = torch.tensor([[1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7],
                        [2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7],
                        [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
                        [5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8],
                        [7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8],
                        [8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9]], dtype=torch.float32)

        # table C RULA-Worksheet
        self.table_C = torch.tensor([[1, 2, 3, 3, 4, 5, 5],
                        [2, 2, 3, 4, 4, 5, 5],
                        [3, 3, 3, 4, 4, 5, 6],
                        [3, 3, 3, 4, 5, 6, 6],
                        [4, 4, 4, 5, 6, 7, 7],
                        [4, 4, 5, 6, 6, 7, 7],
                        [5, 5, 6, 6, 7, 7, 7],
                        [5, 5, 6, 7, 7, 7, 7]], dtype=torch.float32)

        self.angle_dict = {0: 'L_shoulder_Z', 1: 'L_shoulder_line', 2: 'R_shoulder_Z',
                           3: 'R_shoulder_line', 4: 'L_elbow', 5: 'R_elbow',
                           6: 'L_knee', 7: 'R_knee', 8: 'Stoop', 9: 'Trunk_twist',
                           10: 'Trunk_sidebend', 11: 'Neck', 12: 'Neck_sidebend',
                           13: 'Neck_twist'}

        self.score_total = torch.tensor([], requires_grad=True, dtype=torch.float32)

    def compute_scores(self):
        score_upper_arm = torch.tensor([], requires_grad=True, dtype=torch.float32)
        score_lower_arm = torch.tensor([], requires_grad=True, dtype=torch.float32)
        score_trunk = torch.tensor([], requires_grad=True, dtype=torch.float32)
        score_neck = torch.tensor([], requires_grad=True, dtype=torch.float32)
        score_legs = torch.tensor([], requires_grad=True, dtype=torch.float32)

        # Global scores
        score_A = []
        score_B = []
        score_total = []

        angles = self.accumulate_angles()  # This should return a torch tensor
        #print('Angles:', type(angles))


        for i, frame in enumerate(angles):
            #print('--- Frame {}: ---'.format(i))

            # A. Arms Step 1
            max_shoulder = torch.max(torch.abs(frame[0]), torch.abs(frame[2]))
            #print('Max shoulder:', max_shoulder)

            score_val = torch.where(max_shoulder <= 20, torch.tensor(1.0, requires_grad=True), torch.tensor(2.0, requires_grad=True))
            score_val = torch.where((max_shoulder > 20) & (max_shoulder <= 45), torch.tensor(2.0, requires_grad=True), score_val)
            score_val = torch.where((max_shoulder > 45) & (max_shoulder <= 90), torch.tensor(3.0, requires_grad=True), score_val)
            score_val = torch.where(max_shoulder > 90, torch.tensor(4.0, requires_grad=True), score_val)

            score_upper_arm = torch.cat((score_upper_arm, score_val.unsqueeze(0)))

            # A. Check Abduction
            max_abduction = torch.max(frame[1], frame[3])
            #print('Max abduction:', max_abduction)

            # Create a mask for the condition
            #abduction_mask = (max_abduction > 150).float()  # This will be 1.0 if true, 0.0 if false
            abduction_mask = (max_abduction > 150).float().unsqueeze(0)
            score_upper_arm = score_upper_arm + abduction_mask

            # Use the mask to update the score_upper_arm tensor
            # The unsqueeze(0) is necessary to match the dimensions for broadcasting
            #score_upper_arm = score_upper_arm + abduction_mask.unsqueeze(0)
            score_upper_arm = torch.add(score_upper_arm, abduction_mask.unsqueeze(0))

            # A. Arms Step 2
            max_elbow = torch.max(frame[4], frame[5])
            #print('Max elbow:', max_elbow)

            # Use torch.where to handle the conditional logic in a differentiable way
            score_val = torch.where((max_elbow <= 20) | (max_elbow >= 100),
                                    torch.tensor(2.0, requires_grad=True),
                                    torch.tensor(1.0, requires_grad=True))

            # Concatenate the score_val to score_lower_arm
            # Make sure score_lower_arm is a tensor that will have requires_grad=True if necessary
            score_lower_arm = torch.cat((score_lower_arm, score_val.unsqueeze(0)))

            # For indexing, you don't need requires_grad=True because indexing is not a differentiable operation
            # However, ensure that the result of the indexing (curr_score_A) is used in a way that maintains the graph if necessary
            index = ((score_upper_arm[i] - 1) * 3 + (score_lower_arm[i] - 1)).long()
            curr_score_A = self.table_A[index]

            #print('Current Score A:', curr_score_A)
            score_A.append(curr_score_A)

            # B. Neck
            score_val = torch.where(frame[11] >= 40, torch.tensor([2.0], requires_grad=True),
                                    torch.where(frame[11] >= 20, torch.tensor([1.0], requires_grad=True),
                                                torch.where(frame[11] < 20, torch.tensor([4.0], requires_grad=True),
                                                            torch.tensor([3.0], requires_grad=True))))
            score_neck = torch.cat((score_neck, score_val))

            # Update score_neck with side bending condition
            condition = (frame[12] >= 120) | (frame[12] <= 60)
            # Ensure that the update does not break the computational graph
            #score_neck[-1] = score_neck[-1] + condition.float() * torch.tensor(1.0, requires_grad=True)
            #score_neck = torch.cat((score_neck[:-1], score_neck[-1:] + condition.float() * torch.tensor(1.0, requires_grad=True)))
            score_neck = torch.cat((score_neck[:-1], score_neck[-1:] + condition.float() * torch.tensor(1.0, requires_grad=True)))


            #print('Score Neck:', score_neck)

            # Create score_val with requires_grad=True if necessary
            mask1 = (frame[8] <= 15).float()
            mask2 = ((frame[8] > 15) & (frame[8] <= 30)).float()
            mask3 = ((frame[8] > 30) & (frame[8] <= 60)).float()
            mask4 = (frame[8] > 60).float()
            score_val = 1.0 * mask1 + 2.0 * mask2 + 3.0 * mask3 + 4.0 * mask4
            score_val = score_val.unsqueeze(0)  # Add requires_grad=True if this tensor should be part of the gradient computation
            score_trunk = torch.cat((score_trunk, score_val))

            # For score_trunk updates
            additional_score = ((frame[10] <= 60) | (frame[10] >= 120)).float() * torch.tensor(1.0, requires_grad=True)
            additional_score += (frame[9] >= 30).float() * torch.tensor(1.0, requires_grad=True)

            # Now we use the additional_score to update score_trunk without in-place operations
            # Ensure that all tensors involved have requires_grad=True if necessary
            updated_score_trunk = score_trunk[i] + additional_score
            score_trunk = torch.cat((score_trunk[:i], updated_score_trunk.unsqueeze(0), score_trunk[i+1:]))
            #print('Score Trunk:', score_trunk)

            # For score_legs
            min_knee = torch.min(frame[6], frame[7])
            #print('Min knee:', min_knee)

            # Use torch.where to handle the conditional logic in a differentiable way
            score_val = torch.where(min_knee >= 90, torch.tensor(2.0, requires_grad=True), torch.tensor(1.0, requires_grad=True))

            # Concatenate the score_val to score_legs
            score_legs = torch.cat((score_legs, score_val.unsqueeze(0)))
            #print('Score Legs:', score_legs)

            # For indexing, you don't need requires_grad=True because indexing is not a differentiable operation
            # However, ensure that the result of the indexing (curr_score_B) is used in a way that maintains the graph if necessary
            index1 = (score_neck[i] - 1).long()
            index2 = ((score_trunk[i] - 1) * 2 + (score_legs[i] - 1)).long()
            curr_score_B = self.table_B[index1][index2]
            #print('Current Score B:', curr_score_B)
            score_B.append(curr_score_B)

            # Assuming curr_score_A and curr_score_B are tensors that require gradients
            curr_score_A_clamped = torch.clamp(curr_score_A, max=8.0)
            curr_score_B_clamped = torch.clamp(curr_score_B, max=7.0)

            # Indexing to get curr_total
            index1 = (curr_score_A_clamped - 1).long()
            index2 = (curr_score_B_clamped - 1).long()
            curr_total = self.table_C[index1][index2]
            #print('Current Total Score:', curr_total)

            # Append curr_total to score_total
            score_total.append(curr_total)

            # After the loop, convert score_total to a tensor if you need to compute gradients with respect to it
            self.score_total = torch.stack(score_total)  # This will have requires_grad=True if curr_total has it


        self.score_total = torch.stack(score_total)

        return self.score_total


    def accumulate_angles(self):

        """
        computes the angles between body parts as specified by the RULA worksheet
        :return: angles between body parts
        """
        all_angles = []

        for ind in range(len(self.pose_3d)):
            angles_frame = []

            pose = self.transform_pose(self.pose_3d[ind,...].clone()) # deep copy

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

            angles_frame_tensor = torch.tensor(angles_frame, requires_grad=True)
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
        bc = torch.tensor([0.0, 0.0, 1.0], requires_grad=True, device=pose.device)

        cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(cosine_angle)

        angle_degrees = angle * (180 / torch.pi)

        return angle_degrees

    def calculate_angle(self, pose, joint1, joint2, joint3):
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]

        ba = a - b
        bc = c - b

        cosine_angle = torch.dot(ba, bc) / (torch.norm(ba) * torch.norm(bc))
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        return angle * (180.0 / torch.pi)

    def calculate_twist(self, pose, joint1, joint2, joint3, joint4):
        # Initialization with Tensors
        a = pose[:, joint1]
        b = pose[:, joint2]
        c = pose[:, joint3]
        d = pose[:, joint4]

        ba = a - b
        dc = d - c

        # 3. Dot Product and 4. Norm Calculation
        cosine_angle = torch.dot(ba, dc) / (torch.norm(ba) * torch.norm(dc))

        # 6. Angle Calculation and 8. Handling Numerical Instabilities
        angle = torch.acos(torch.clamp(cosine_angle, min=-1.0, max=1.0))

        # 7. Degree Conversion
        return angle * (180.0 / torch.pi)

#%%

#%%
