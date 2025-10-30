import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision.utils import save_image
from tqdm import tqdm

config = {
    "clip_model_name": "openai/clip-vit-base-patch32",
    "output_dir": "diffusion_model/saved_models/",
    "image_size": 32,
    "embedding_dim": 512,
}

inference_config = {
    "model_epoch_to_load": 150,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "num_images_per_prompts": 64,
    "output_grid_dir": "diffusion_model/saved_images/cifar10_grids/",
    "mixed_precision_dtype": torch.bfloat16,
    "seed": 0
}

CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
@torch.no_grad()
def get_text_embeddings(prompts, tokenizer, text_encoder, device, dtype):
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state
    return text_embeddings.to(dtype)

@torch.no_grad()
def get_uncond_embeddings(tokenizer, text_encoder, device, dtype):
    return get_text_embeddings(
        [""] * inference_config["num_images_per_prompts"], 
        tokenizer, 
        text_encoder, 
        device, 
        dtype
    )

def main():
    os.makedirs(inference_config["output_grid_dir"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = inference_config["mixed_precision_dtype"]

    set_seed(inference_config["seed"])

    tokenizer = CLIPTokenizer.from_pretrained(config["clip_model_name"])
    text_encoder = CLIPTextModel.from_pretrained(config["clip_model_name"])
    text_encoder.to(device, dtype).eval()

    model_path = os.path.join(
        config["output_dir"],
        f"epoch_{inference_config['model_epoch_to_load']}"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=dtype)
    unet.to(device).eval()

    unet = torch.compile(unet, mode="reduce-overhead")

    scheduler = DDIMScheduler.from_pretrained(model_path)
    scheduler.set_timesteps(inference_config["num_inference_steps"])
    scheduler.timesteps = scheduler.timesteps.to(device) 

    uncond_embeddings = get_uncond_embeddings(tokenizer, text_encoder, device, dtype) # (num_images_per_prompts)

    conditional_prompts = [f"a photo of a {label}" for label in CIFAR10_LABELS]
    all_cond_embeddings = get_text_embeddings(
        conditional_prompts, 
        tokenizer, 
        text_encoder, 
        device, 
        dtype
    )

    generator = torch.Generator(device=device).manual_seed(inference_config["seed"])

    for i, label_name in enumerate(tqdm(CIFAR10_LABELS, desc="Generating images for all labels")):
        cond_embeddings = all_cond_embeddings[i].unsqueeze(0).repeat(
            inference_config["num_images_per_prompts"], 1, 1
        )

        combined_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)

        latents = torch.randn(
            (
                inference_config["num_images_per_prompts"],
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

                noise_pred = noise_pred_uncond + inference_config["guidance_scale"] * (noise_pred_cond - noise_pred_uncond)

                latents = scheduler.step(noise_pred, t, latents).prev_sample

        images = (latents / 2.0 + 0.5).clamp(0, 1)

        images = images.cpu().to(torch.float32)

        save_path = os.path.join(
            inference_config["output_grid_dir"],
            f"{label_name}.png"
        )
        save_image(
            images, 
            fp=save_path,
            nrow=8
        )

if __name__ == "__main__":
    main()