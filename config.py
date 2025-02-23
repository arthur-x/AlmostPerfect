import os
import torch


def clean_directory(directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file():
                os.remove(entry.path)

sample_path = "samples/"
score_path = sample_path + "scores.json"
model_path = "models/"
checkpoint_path = model_path + "checkpoint"
reward_model_path = model_path + "reward_model.pth"
lora_path = "outputs/"
lora_rank = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generate_sample_size = 1000
prompt = "a neon sign"
negative_prompt = "worst quality, low quality"
height = 1080
width = 720
num_inference_steps = 20
guidance_scale = 5
eta = 1
train_reward_num_epochs = 10
train_reward_batch_size = 100
train_reward_test_split_ratio = 0.1
train_reward_learning_rate = 1e-3
train_policy_num_epochs = 50
train_save_freq = 10
train_sample_batch_size = 4
train_sample_num_batches_per_epoch = 4
train_policy_batch_size = 1
train_policy_learning_rate = 1e-4
train_adv_clip_max = 2
train_clip_range = 1e-4
