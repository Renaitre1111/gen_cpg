import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import math

CPG_CONFIG = {
    "dataset": "cifar10",
    "num_classes": 10,
    "num_labels": 400,
    "imb_factor_l": 100,
    "data_dir": "./data",
    "lb_idx_path": "diffusion_model/label_idx/cifar10/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx/lb_labels_400_100_4600_100_exp_random_noise_0.0_seed_1_idx.npy",
}

DIFFUSION_CONFIG = {
    "clip_model_name": "openai/clip-vit-base-patch32",
    "output_dir": "diffusion_model/saved_models/epoch150",
    "image_size": 32,
    "embedding_dim": 512,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "mixed_precision_dtype": torch.bfloat16,
    "generation_batch_size": 256,
    "seed": 1
}

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]


def get_text_embeddings(prompts, tokenizer, text_encoder, device, dtype):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
    return text_embeddings.to(dtype)

@torch.no_grad()
def gen_all_cond_embeddings(label_names, tokenizer, text_encoder, device, dtype):
    conditional_prompts = [f"a photo of a {label_name}" for label_name in label_names]
    all_cond_embeddings = get_text_embeddings(
        conditional_prompts, 
        tokenizer, 
        text_encoder, 
        device, 
        dtype
    )
    return all_cond_embeddings


@torch.no_grad()
def gen_samples_for_class(label_name, num_to_gen, unet, scheduler, cond_embedding, uncond_embedding, generator, device, dtype, config):
    print(f"Generating {num_to_gen} samples for {label_name}")
    cond_embedding.unsqueeze(0).repeat(num_to_gen, 1, 1)
    uncond_embedding.unsqueeze(0).repeat(num_to_gen, 1, 1)

    combined_embeddings = torch.cat([uncond_embedding, cond_embedding], dim=0)
    latents = torch.randn(
            (
                num_to_gen,
                unet.config.in_channels,
                config["image_size"],
                config["image_size"]
            ),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    
    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps, leave=False, desc=f"Sampling {label_name}"):
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=combined_embeddings,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            noise_pred = noise_pred_uncond + config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    images = (latents / 2.0 + 0.5).clamp(0, 1)
    images = images.cpu().to(torch.float32)
    
    return images