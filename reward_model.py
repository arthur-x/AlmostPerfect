import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from config import *


class LatentRewardDataset(Dataset):
    def __init__(self, latents, rewards):
        assert len(latents) == len(rewards), "Latents and rewards must have the same length"
        self.latents = latents
        self.rewards = rewards

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        reward = self.rewards[idx]
        return {'latent': latent, 'reward': reward}


class LatentRewardCNN(nn.Module):
    def __init__(self):
        super(LatentRewardCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        reward = self.fc(x)
        return reward


class Scorer:
    def __init__(self, vae):
        self.reward_model = LatentRewardCNN().to(device)
        if os.path.exists(reward_model_path):
            reward_model_pretrained = torch.load(reward_model_path, weights_only=True)
            self.reward_model.load_state_dict(reward_model_pretrained['model_state_dict'])
            print("Loaded existing reward model.")
        else:
            print("Initialized a new reward model.")
        self.vae = vae
        self.preprocess = transforms.Compose([
            transforms.Normalize([0.5], [0.5])  # Normalize to range [-1, 1]
        ])

    def score(self, images):
        image_tensor = self.preprocess(images).to(device).half()  # Ensure consistent float16 input
        if len(image_tensor.shape) == 3: image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.mode().float()  # Cast back to float32
            rewards_predicted = self.reward_model(latents)
        return rewards_predicted, latents
