import os
import numpy as np
from utils import Setup_environment
Setup_environment()
from src.hpe_wrapper import Wrapper_2Dpose, Wrapper_3Dpose
from torch.autograd import gradcheck
from src.ergonomics_torch import Ergonomics_Torch
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from visualiser import visualize_poses_in_video
from HumanPoseGAN import HumanPoseDiscriminator
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
data_3d_path = 'data/intermedia/data_3d.npz'
# Check if the data_3d file exists
if os.path.exists(data_3d_path):
    # Load the data_3d if it exists
    loaded_data = np.load(data_3d_path)
    data_3d = loaded_data['data_3d']
else:
    model_2D ='./detectron/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
    weights_2D = './detectron/checkpoint/model_final_997cc7.pkl'
    model_3D = './VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin'

    pose2d = Wrapper_2Dpose(model=model_2D, weights=weights_2D, ROI_thr=0.75)
    pose_3d = Wrapper_3Dpose(model_3D)
    config = load_config()
    print(config['video_object'])

    video_object = Video_wrapper(config['video_object'], resize_video_by=0.3)
    data_2d, metadata_vid = pose2d.predict_2D_poses(input_video_object=video_object)
    data_3d = pose_3d.predict_3D_poses(data_2d, metadata_vid)
    np.savez(data_3d_path, data_3d=data_3d)

def print_grad(named_tensor):
    tensor_name, tensor = named_tensor
    if tensor.grad is not None:
        print(f"Gradient for {tensor_name}: {tensor.grad}")
    else:
        print(f"No gradient for {tensor_name}")

def vector_difference(vector1, vector2):
    vector2 = vector2.cpu().numpy()
    vector1 = np.array(vector1)

    difference = vector1 - vector2

    return difference

def matrix_difference(matrix1, matrix2):
    if isinstance(matrix2, torch.Tensor):
        matrix2 = matrix2.cpu().numpy()
    elif not isinstance(matrix2, np.ndarray):
        matrix2 = np.array(matrix2)
    difference = matrix1 - matrix2

    # calculate the Euclidean norm (L2 norm) of the difference
    norm = np.linalg.norm(difference)

    return norm

def TMSE(optimized_poses):
    diffs = optimized_poses[1:] - optimized_poses[:-1]
    loss = torch.mean(diffs.pow(2))
    return loss

def optimize_pose(pose_3d_initial, ergonomic_torch, discriminator, file, lr=0.01, num_steps=100, print_interval=20):

    pose_3d = pose_3d_initial.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([pose_3d], lr=lr)
    #optimizer = optim.SGD([pose_3d], lr=lr)
    #optimizer = optim.RMSprop([pose_3d], lr=lr)
    # Coefficients for loss
    alpha = torch.tensor(0.4, dtype=torch.float32, requires_grad=True, device=device)  # Weight for ergo loss
    beta = torch.tensor(0.4, dtype=torch.float32, requires_grad=True, device=device)   # Weight for structural loss
    gamma = torch.tensor(0.2, dtype=torch.float32, requires_grad=True) # Weight for Discriminator loss

    for step in range(num_steps):
        print(step, '--------------------------------')

        optimizer.zero_grad()

        # cart_loss = L1_loss(pose_3d, pose_3d_initial)
        # print('cart_loss: ', cart_loss)
        ergo_loss = ergonomic_torch(pose_3d)
        structural_loss = TMSE(pose_3d)
        print('ergo_loss: ', ergo_loss)
        print('structural_loss: ', structural_loss)

        # Discriminator loss
        discriminator_losses = []
        for frame in pose_3d:
            frame_loss = -torch.log(discriminator(frame.flatten()))
            discriminator_losses.append(frame_loss)

        # Aggregate discriminator losses
        discriminator_loss = torch.sum(torch.stack(discriminator_losses))

        print('discriminator_loss: ', discriminator_loss.item())

        # Compute the composite loss
        composite_loss = alpha * ergo_loss + beta * structural_loss + gamma * discriminator_loss

        loss = torch.sum(composite_loss)

        loss.backward()
        print('pose_3d.grad =', pose_3d.grad)
        print('pose_3d.requires_grad', pose_3d.requires_grad)

        optimizer.step()

        if step % print_interval == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    visualize_poses_in_video(data_3d, pose_3d.detach(), alpha.item(), beta.item(), gamma.item(), optimizer.__class__.__name__, file)

    return pose_3d.detach()

if __name__ == '__main__':
    npz_files = glob.glob('data/npz/train/*.npz')

    for file in npz_files:
        print(f"Processing {file}...")
        loaded_data = np.load(file)
        initial_pose_3d = loaded_data['kps']

        if isinstance(initial_pose_3d, np.ndarray):
            initial_pose_3d = torch.from_numpy(initial_pose_3d).float()
        initial_pose_3d = initial_pose_3d.to(device)
        initial_pose_3d.requires_grad_(True)

        # HumanPose GAN
        discriminator = HumanPoseDiscriminator().to(device)
        discriminator.load_state_dict(torch.load('models/discriminator.pth', map_location=device))
        discriminator.eval()

        optim_module = Ergonomics_Torch(device)
        optim_module.to(device)

        optimized_pose = optimize_pose(initial_pose_3d, optim_module, discriminator, file)
        #difference = matrix_difference(initial_pose_3d[1, :, :], optimized_pose[1, :, :])
        #print("Difference between the matrix:")
        #print(difference)
    """initial_pose_3d = data_3d

    if isinstance(initial_pose_3d, np.ndarray):
        initial_pose_3d = torch.from_numpy(initial_pose_3d).float()
    initial_pose_3d = initial_pose_3d.to(device)
    initial_pose_3d.requires_grad_(True)

    # HumanPose GAN
    discriminator = HumanPoseDiscriminator().to(device)
    discriminator.load_state_dict(torch.load('models/discriminator.pth', map_location=device))
    discriminator.eval()

    optim_module = Ergonomics_Torch(device)
    optim_module.to(device)

    optimized_pose = optimize_pose(initial_pose_3d, optim_module, discriminator)
    difference = matrix_difference( data_3d[1,:,], optimized_pose[1,:,:])
    print("Difference between the matrix:")
    print(difference)
    optimized_pose = optimized_pose.cpu()
    optimized_pose = optimized_pose.numpy()"""