from torch import autocast, GradScaler
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from diffusers.utils import (convert_state_dict_to_diffusers,
                             convert_all_state_dict_to_peft,
                             convert_state_dict_to_kohya,
                             convert_unet_state_dict_to_peft)
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from config import *
from reward_model import Scorer
from utils.ddim_with_logprob import ddim_step_with_logprob
from utils.pipeline_with_logprob import pipeline_with_logprob


if __name__ == "__main__":
    pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_path)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    pipeline.unet.to(device, dtype=torch.float16)
    pipeline.vae.to(device, dtype=torch.float16)
    pipeline.text_encoder.to(device, dtype=torch.float16)

    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        print("Enabled xformers_memory_efficient_attention.")
        pipeline.unet.enable_xformers_memory_efficient_attention()

    unet_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    pipeline.unet.add_adapter(unet_lora_config)
    for param in pipeline.unet.parameters():
        # only upcast trainable parameters (LoRA) into fp32
        if param.requires_grad:
            param.data = param.to(torch.float32)
    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=train_policy_learning_rate)
    if os.path.exists(lora_path + f"diffusers_lora_epoch_{train_policy_num_epochs}.safetensors"):
        lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(lora_path + f"diffusers_lora_epoch_{train_policy_num_epochs}.safetensors")
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        set_peft_model_state_dict(pipeline.unet, unet_state_dict, adapter_name="default")
        print("Loaded existing LoRA.")
        optimizer.load_state_dict(torch.load(lora_path + "optimizer.pth"))
    else:
        print("Initialized a new LoRA.")

    clean_directory(lora_path)

    scaler = GradScaler()

    scorer = Scorer(pipeline.vae)

    prompt_embeds, negative_prompt_embeds = pipeline.encode_prompt(
        prompt=[prompt],
        negative_prompt=[negative_prompt],
        do_classifier_free_guidance=True,
        device=device,
        num_images_per_prompt=1
    )

    sample_prompt_embeds = prompt_embeds.repeat(train_sample_batch_size, 1, 1)
    sample_neg_prompt_embeds = negative_prompt_embeds.repeat(train_sample_batch_size, 1, 1)
    train_prompt_embeds = prompt_embeds.repeat(train_policy_batch_size, 1, 1)
    train_neg_prompt_embeds = negative_prompt_embeds.repeat(train_policy_batch_size, 1, 1)

    samples_per_epoch = (train_sample_batch_size * train_sample_num_batches_per_epoch)
    assert samples_per_epoch > train_policy_batch_size
    assert samples_per_epoch % train_policy_batch_size == 0

    for epoch in range(1, train_policy_num_epochs+1):
        #################### SAMPLING ####################
        pipeline.unet.eval()
        samples = []
        for i in tqdm(
            range(train_sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            dynamic_ncols = True,
            leave=False
        ):

            # sample
            images, _, latents, log_probs = pipeline_with_logprob(
                pipeline,
                prompt_embeds=sample_prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                eta=eta,
                output_type="pt",
                height=height,
                width=width
            )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(train_sample_batch_size, 1)
            rewards = scorer.score(images)[0]

            samples.append(
                {
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        rewards = samples["rewards"]
        print(f"Epoch {epoch}: avg_reward: {rewards.mean().item():.4f}")
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        samples["advantages"] = advantages
        del samples["rewards"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        samples_batched = {
            k: v.reshape(-1, train_policy_batch_size, *v.shape[1:])
            for k, v in samples.items()
        }
        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]

        #################### TRAINING ####################
        pipeline.unet.train()
        for i, sample in tqdm(
            list(enumerate(samples_batched)),
            desc=f"Epoch {epoch}: training",
            dynamic_ncols = True,
            leave=False
        ):
            embeds = torch.cat(
                [train_neg_prompt_embeds, train_prompt_embeds]
            )
            for j in tqdm(
                range(num_inference_steps),
                desc="Timestep",
                dynamic_ncols = True,
                leave=False
            ):
                with autocast(device_type="cuda"):
                    noise_pred = pipeline.unet(
                        torch.cat([sample["latents"][:, j]] * 2),
                        torch.cat([sample["timesteps"][:, j]] * 2),
                        embeds,
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale
                        * (noise_pred_text - noise_pred_uncond)
                    )
                    # compute the log prob of next_latents given latents under the current model
                    _, log_prob = ddim_step_with_logprob(
                        pipeline.scheduler,
                        noise_pred,
                        sample["timesteps"][:, j],
                        sample["latents"][:, j],
                        eta=eta,
                        prev_sample=sample["next_latents"][:, j],
                    )

                    # ppo logic
                    advantages = torch.clamp(
                        sample["advantages"],
                        -train_adv_clip_max,
                        train_adv_clip_max,
                    )
                    ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - train_clip_range,
                        1.0 + train_clip_range,
                    )
                    loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  # we won't update scaler often enough to trigger growth, so we don't save it
        optimizer.zero_grad()

        if epoch % train_save_freq == 0:
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(pipeline.unet))
            StableDiffusionPipeline.save_lora_weights(
                save_directory=lora_path,
                unet_lora_layers=unet_lora_state_dict,
                weight_name=f"diffusers_lora_epoch_{epoch}.safetensors"
            )

            # Convert to WebUI/ComfyUI format
            lora_state_dict = load_file(lora_path + f"diffusers_lora_epoch_{epoch}.safetensors")
            peft_state_dict = convert_all_state_dict_to_peft(lora_state_dict)
            kohya_state_dict = convert_state_dict_to_kohya(peft_state_dict)
            save_file(kohya_state_dict, lora_path + f"ui_lora_epoch_{epoch}.safetensors")

            torch.save(optimizer.state_dict(), lora_path + "optimizer.pth")
