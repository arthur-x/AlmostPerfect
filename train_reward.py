from tqdm import tqdm
import json
from PIL import Image
from reward_model import LatentRewardDataset, Scorer
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from config import *


if __name__ == "__main__":
    with open(score_path, "r") as f:
        scores = json.load(f)
    filenames = list(scores.keys())
    s = 0
    for f in filenames:
        s += scores[f]
    score_size = len(filenames)
    print("avg_score:", s / score_size)

    latent_data = torch.zeros((score_size, 4, height//8, width//8))
    reward_data = torch.zeros((score_size, 1))
    scorer = Scorer(AutoencoderKL.from_pretrained(checkpoint_path + "/vae", torch_dtype=torch.float16).to(device))

    for i in tqdm(
            range(0, score_size),
            desc=f"Preprocessing",
            dynamic_ncols=True
    ):
        if os.path.exists(sample_path + filenames[i]):
            image = Image.open(sample_path + filenames[i]).convert("RGB")
            latent_data[i] = scorer.score(ToTensor()(image))[1]
            reward_data[i] = scores[filenames[i]]

    dataset = LatentRewardDataset(latent_data, reward_data)
    test_size = int(len(dataset) * train_reward_test_split_ratio)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=train_reward_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=train_reward_batch_size, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(scorer.reward_model.parameters(), lr=train_reward_learning_rate)

    for epoch in range(1, train_reward_num_epochs+1):
        scorer.reward_model.train()
        epoch_loss = 0
        for batch in train_loader:
            latent_batch = batch["latent"].to(device)
            reward_batch = batch["reward"].to(device)
            reward_predicted = scorer.reward_model(latent_batch)
            loss = criterion(reward_predicted, reward_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)

        scorer.reward_model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                latent_batch = batch["latent"].to(device)
                reward_batch = batch["reward"].to(device)
                reward_predicted = scorer.reward_model(latent_batch)
                loss = criterion(reward_predicted, reward_batch)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch}: "
              f"Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    print("Training completed.")
    torch.save({"model_state_dict": scorer.reward_model.state_dict()}, reward_model_path)
    print(f"Reward model saved.")
