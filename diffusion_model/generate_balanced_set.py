import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torchvision
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
    "balanced_data_path": "./data/cifar10/balanced_data_400_100_4600_100_exp_random_noise_0.0_seed_1.npz"
}

DIFFUSION_CONFIG = {
    "clip_model_name": "openai/clip-vit-base-patch32",
    "output_dir": "diffusion_model/saved_models",
    "model_epoch_to_load": 150,
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
    return get_text_embeddings([""], tokenizer, text_encoder, device, dtype)

@torch.no_grad()
def gen_samples_for_class(
    label_name,
    num_to_gen,
    unet,
    scheduler,
    cond_embedding,
    uncond_embedding,
    generator,
    device,
    dtype,
    config,
):
    print(f"Generating {num_to_gen} samples for {label_name}")
    if num_to_gen == 0:
        return torch.empty((0, unet.config.in_channels, config["image_size"], config["image_size"]), device=device, dtype=dtype).cpu().to(torch.float32).numpy()

    bs = int(config.get("generation_batch_size", 64))
    num_batches = (num_to_gen + bs - 1) // bs

    all_imgs = []
    use_cuda_autocast = (isinstance(device, str) and device.startswith("cuda")) or (device == torch.device("cuda"))

    for b in range(num_batches):
        start = b * bs
        end = min(num_to_gen, (b + 1) * bs)
        cur_bs = end - start

        cond_emb = cond_embedding.unsqueeze(0).repeat(cur_bs, 1, 1)
        uncond_emb = uncond_embedding.repeat(cur_bs, 1, 1)
        combined_embeddings = torch.cat([uncond_emb, cond_emb], dim=0).to(device=device, dtype=dtype)

        latents = torch.randn(
            (cur_bs, unet.config.in_channels, config["image_size"], config["image_size"]),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = latents * scheduler.init_noise_sigma

        for t in tqdm(scheduler.timesteps, leave=False, desc=f"Sampling {label_name} (batch {b+1})"):
            if use_cuda_autocast:
                ctx = torch.autocast("cuda", dtype=dtype)
            else:
                ctx = torch.cuda.amp.autocast(enabled=False)
            with ctx:
                latent_in = torch.cat([latents] * 2)
                latent_in = scheduler.scale_model_input(latent_in, t)

                noise_pred = unet(
                    latent_in,
                    t,
                    encoder_hidden_states=combined_embeddings,
                ).sample

                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise = noise_uncond + config["guidance_scale"] * (noise_cond - noise_uncond)

                latents = scheduler.step(noise, t, latents).prev_sample

        images = (latents / 2.0 + 0.5).clamp(0, 1).cpu().to(torch.float32)
        all_imgs.append(images)

        del latents, cond_emb, uncond_emb, combined_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    imgs = torch.cat(all_imgs, dim=0)[:num_to_gen]
    return imgs.numpy()

def group_imgs_by_label(imgs, labels, classes):
    group = []
    for lb in range(len(classes)):
        group.append(imgs[labels == lb])
    return group

def get_imbalanced_data(dataset, lb_idx):
    if dataset == "cifar10":
        all_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=None
        )
    elif dataset == "cifar100":
        all_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=None
        )

    lb_imgs = all_dataset.data[lb_idx]
    lb_targets = np.array(all_dataset.targets)[lb_idx]
    classes = all_dataset.classes
    
    group = group_imgs_by_label(lb_imgs, lb_targets, classes)

    counts = np.bincount(lb_targets, minlength=len(classes))
    max_count = np.max(counts)
    num_to_gens = max_count - counts

    return group, num_to_gens, classes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = DIFFUSION_CONFIG["mixed_precision_dtype"]

    set_seed(DIFFUSION_CONFIG["seed"])

    lb_idx = np.load(CPG_CONFIG["lb_idx_path"])
    group, num_to_gens, classes = get_imbalanced_data(CPG_CONFIG["dataset"], lb_idx)


    tokenizer = CLIPTokenizer.from_pretrained(DIFFUSION_CONFIG["clip_model_name"])
    text_encoder = CLIPTextModel.from_pretrained(DIFFUSION_CONFIG["clip_model_name"])
    text_encoder.to(device, dtype).eval()

    model_path = os.path.join(
        DIFFUSION_CONFIG["output_dir"],
        f"epoch_{DIFFUSION_CONFIG['model_epoch_to_load']}"
    )

    unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=dtype)
    unet.to(device).eval()
    
    # unet = torch.compile(unet, mode="reduce-overhead")

    scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(DIFFUSION_CONFIG["num_inference_steps"])
    scheduler.timesteps = scheduler.timesteps.to(device) 

    uncond_embedding = get_uncond_embeddings(tokenizer, text_encoder, device, dtype)

    conditional_prompts = [f"a photo of a {label_name}" for label_name in classes]
    all_cond_embeddings = get_text_embeddings(
        conditional_prompts, 
        tokenizer, 
        text_encoder, 
        device, 
        dtype
    )
    generator = torch.Generator(device=device).manual_seed(DIFFUSION_CONFIG["seed"])

    for i, label in enumerate(classes):
        imgs = gen_samples_for_class(label, num_to_gens[i], 
                              unet, scheduler, 
                              all_cond_embeddings[i]
                              , uncond_embedding, 
                              generator, device, dtype, DIFFUSION_CONFIG)
        imgs = (np.transpose(imgs, (0, 2, 3, 1)) * 255.0).round().clip(0, 255).astype(np.uint8)

        group[i] = np.concatenate([group[i], imgs], axis=0)

    balanced_imgs = np.concatenate(group, axis=0)  # (N_total, 32, 32, 3), uint8
    balanced_labels = np.concatenate([np.full(len(group[i]), i, dtype=np.int64) for i in range(len(group))])

    np.savez(CPG_CONFIG["balanced_data_path"], imgs=balanced_imgs, labels=balanced_labels)
    

if __name__ == "__main__":
    main()