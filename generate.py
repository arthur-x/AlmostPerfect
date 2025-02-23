from tqdm import tqdm
import random
from diffusers import StableDiffusionPipeline
from config import *


if __name__ == "__main__":
    clean_directory(sample_path)
    pipeline = StableDiffusionPipeline.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
    if os.path.exists(lora_path + f"diffusers_lora_epoch_{train_policy_num_epochs}.safetensors"):
        pipeline.load_lora_weights(lora_path, weight_name=f"diffusers_lora_epoch_{train_policy_num_epochs}.safetensors")
        print("Generating with LoRA.")
    else:
        print("Generating without LoRA.")
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    seed_set = set()
    for i in tqdm(
            range(generate_sample_size),
            desc=f"Generating",
            dynamic_ncols=True
    ):
        while (seed := random.randrange(100000, 1000000)) in seed_set:
            continue
        seed_set.add(seed)
        image = pipeline(prompt=prompt,
                         negative_prompt=negative_prompt,
                         width=width,
                         height=height,
                         num_inference_steps=num_inference_steps,
                         guidance_scale=guidance_scale,
                         eta=eta,
                         generator=torch.manual_seed(seed)).images[0]
        image.save(sample_path + f"{seed}.png")
