import torch
from torch import Tensor
import tqdm

@torch.no_grad()
def encode(pipe, images: Tensor, device):
    images_scaled = images.to(device) * 2 - 1  # scale to [-1, 1]
    latents = pipe.vae.encode(images_scaled)
    l = 0.18215 * latents.latent_dist.sample()  # empriical value for ensure the latent has unit variance.
    return l

@torch.no_grad()
def sample(
    pipe,
    prompt,
    start_step=0,
    start_latents=None,
    guidance_scale=3.5,
    num_inference_steps=30,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
    prefix="",
):
    batch_size = num_images_per_prompt if start_latents is None else start_latents.shape[0]
    # encode prompt
    text_emb = pipe._encode_prompt(
        prompt, device, batch_size, do_classifier_free_guidance, negative_prompt
    )
    
    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # create a random starting point if we don't have one
    if start_latents is None:
        start_latents = torch.randn(batch_size, 4, 64, 64, device=device)  # (1, 4, 64, 64) is the latent shape for diffusion models. (4 is the number of dimensions of the encoder)
        start_latents *= pipe.scheduler.init_noise_sigma
    
    latents = start_latents.clone()
    
    if prefix is not None:
        print(f"{prefix}: Sampling Images for {num_inference_steps} steps...")
    
    for i in tqdm.tqdm(range(start_step, num_inference_steps)):
        
        t = pipe.scheduler.timesteps[i]
        
        # expand latents for classifier guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents  # (2*B, C, H, W)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t) 
        
        # Predict the noise
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample  # (2*B, C, H, W)
        
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)  # (B, C, H, W), (B, C, H, W)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        prev_t = pipe.scheduler.timesteps[i+1] if i+1 < num_inference_steps else pipe.scheduler.timesteps[-1]  # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt() * noise_pred
        latents = alpha_t_prev.sqrt() * predicted_x0 + direction_pointing_to_xt
    
    # last generative step
    images = pipe.decode_latents(latents)  # pass stable diffusion decoder
    
    return images


@torch.no_grad()
def invert(
    pipe,
    start_latents, 
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    negative_prompt="",
    device="cuda",
    prefix="",
):
    # import pdb; pdb.set_trace()
    batch_size = start_latents.shape[0]
    
    # encode prompt, #TODO: implement per image prompt encoding, currently it is same for all images.
    text_emb = pipe._encode_prompt(
        prompt, device, batch_size, do_classifier_free_guidance, negative_prompt
    )
    
    latents = start_latents.clone()
    
    # TODO: why keep intermediate latents
    intermediate_latents = []
    
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    timesteps = reversed(pipe.scheduler.timesteps)
    
    if prefix is not None:
        print(f"{prefix}: Encoding Images for {num_inference_steps} steps...")
        
    for i in tqdm.tqdm(range(1, num_inference_steps), total=num_inference_steps-1):
        #TODO: why skip final iteration, why start from 1.
        if i >= num_inference_steps - 1: continue
        
        t = timesteps[i]
        
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_emb).sample  # (2*B, C, H, W)
        
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)  # (B, C, H, W), (B, C, H, W)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
        current_t = max(0, t.item() - (1000//num_inference_steps))  # TODO: why?
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]
        
        latents = (latents - (1-alpha_t).sqrt() * noise_pred) * (alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt() * noise_pred
        
        intermediate_latents.append(latents)
        
    return  intermediate_latents