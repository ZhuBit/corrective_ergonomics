import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn as nn
import random
import plotly.graph_objects as go
import webbrowser
import time
class HumanPoseDiscriminator(nn.Module):
    def __init__(self):
        super(HumanPoseDiscriminator, self).__init__()
        # (17 keypoints * 3 coordinates)
        self.model = nn.Sequential(
            nn.Linear(51, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, frame):
        validity = self.model(frame)
        return validity

class HumanPoseGenerator(nn.Module):
    def __init__(self):
        super(HumanPoseGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 51), # 17 keypoints * 3 coordinates
            nn.Tanh()
        )

    def forward(self, z):
        pose = self.model(z)
        return pose
class HPFrameDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.npz')]
        self.transform = transform
        self.frames = []
        # get all frames from all files
        for file_path in self.file_paths:
            data = np.load(file_path)['kps']
            for frame in data:
                self.frames.append(frame.reshape(-1))
        print("Loaded {} frames".format(len(self.frames)))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sample = self.frames[idx]
        label = torch.zeros(1, dtype=torch.float32)

        """if create_negative_samples:
            # 1. create negative samples by randomly permuting the keypoints
            # 2. create negative samples by combing keypoints from different frames
            label = torch.ones(1, dtype=torch.float32)
            sample = self.generate_negative_sample(sample)"""

        if self.transform:
            sample = self.transform(sample)

        return torch.tensor(sample, dtype=torch.float32), label

def visualize_frame(frame):
    x = frame[:, 0]
    y = frame[:, 2]
    z = - frame[:, 1]

    scatter = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(size=5, color='red')
    )

    pairs = [(0,1), (0,4), (0,7), (7,8), (8,9), (9,10), (4,5), (1,2), (5,6), (2,3),
             (8,11), (8,14), (11,12), (14,15), (12,13), (15,16)]

    lines = []
    for pair in pairs:
        lines.append(
            go.Scatter3d(
                x=[x[pair[0]], x[pair[1]]],
                y=[y[pair[0]], y[pair[1]]],
                z=[z[pair[0]], z[pair[1]]],
                mode='lines',
                line=dict(color='green', width=5)
            )
        )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(nticks=10, range=[min(x), max(x)]),
            yaxis=dict(nticks=10, range=[min(y), max(y)]),
            zaxis=dict(nticks=10, range=[min(z), max(z)])
        )
    )

    fig = go.Figure(data=[scatter] + lines, layout=layout)

    filename = "plot.html"
    fig.write_html(filename)

    webbrowser.open(filename)
    time.sleep(3)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data_path = 'data/npz/train'

    batch_size = 64
    num_epochs = 1000

    total_samples = 0
    correct_predictions = 0
    true_positives = 0
    total_predicted_positives = 0

    train_dataset = HPFrameDataset(train_data_path)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset = HPFrameDataset(test_data_path)


    generator = HumanPoseGenerator().to(device)
    discriminator = HumanPoseDiscriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    adversarial_loss = torch.nn.BCELoss()

    for epoch in range(num_epochs):
        print("Epoch: {}".format(epoch))

        for i, (real_samples, _) in enumerate(dataloader):
            print("Batch: {}".format(i))
            real_samples = real_samples.to(device)
            valid = torch.ones((real_samples.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((real_samples.size(0), 1), requires_grad=False).to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(real_samples.shape[0], 100).to(device)
            generated_samples = generator(z)
            if epoch % 100 == 0 and i == 0:
                visualize_frame(generated_samples[0].cpu().detach().numpy().reshape(-1, 3))
            g_loss = adversarial_loss(discriminator(generated_samples), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_samples), valid)
            fake_loss = adversarial_loss(discriminator(generated_samples.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # Calculate predictions for accuracy and precision
            with torch.no_grad():
                real_predictions = discriminator(real_samples)
                fake_predictions = discriminator(generated_samples.detach())

                # Threshold predictions for binary classification
                real_predictions = real_predictions > 0.5
                fake_predictions = fake_predictions > 0.5

                total_samples += real_samples.size(0) * 2  # real and fake samples
                correct_predictions += (real_predictions == valid).sum().item() + (fake_predictions == fake).sum().item()
                true_positives += real_predictions.sum().item()
                total_predicted_positives += real_predictions.sum().item() + fake_predictions.sum().item()

            accuracy = correct_predictions / total_samples
            precision = true_positives / total_predicted_positives if total_predicted_positives > 0 else 0

            print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}] [Accuracy: {accuracy:.2f}] [Precision: {precision:.2f}]")

    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')