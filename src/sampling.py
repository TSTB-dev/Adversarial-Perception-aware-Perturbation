import torch
from torch import Tensor
import tqdm

from sampling_functions import encode, sample, invert

def ddim_sampling(
    pipe,
    images: Tensor,
    prompt: str,
    num_inference_steps: int,
    device: str,
    num_samples: int=1,
    ddim_sampling_step=None,
    guidance_scale: float=3.5,
    do_classifier_free_guidance: bool=True,
    negative_prompt: str="",
    prefix="",
):
    if ddim_sampling_step is None:
        ddim_sampling_step = 0
    
    # encode image to latent space
    start_latents = encode(pipe, images, device=device)  # (B, c, h, w)  
    
    # invert latent space to image space. 
    inverted_latents = invert(
        pipe=pipe,
        start_latents=start_latents,
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_samples,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        device=device,
        prefix=prefix,
    )  # (n-2, B, c, h, w)
    
    if ddim_sampling_step < num_inference_steps:
        inverted_latents = inverted_latents[-(ddim_sampling_step + 1)]
    else:
        inverted_latents = inverted_latents[-1]
        
    # sample from inverted latent space
    samples = sample(
        pipe=pipe,
        prompt=prompt,
        start_latents=inverted_latents,
        start_step=ddim_sampling_step,
        num_inference_steps=num_inference_steps,
        prefix=prefix,
    )  # (B, c, h, w)
    
    return samples
    
    
    
    