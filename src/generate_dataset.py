import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
from pathlib import Path

import time
import random
import numpy as np
import torch
import cv2
import h5py
from tqdm.auto import tqdm
from accelerate import PartialState
from torch.utils.data import DataLoader
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
import warnings
warnings.filterwarnings("ignore")

from dataset import PetsDataset
from sampling import ddim_sampling

def get_diffusion_pipe(model_name, device):    
    print(f"Loading model: {model_name}")
    pipe = StableDiffusionPipeline.from_pretrained(model_name).to(device)
    return pipe

def postprocess(images, image_size):
    images = [(img * 255.0).astype(np.uint8) for img in images]
    images = [cv2.resize(img, (image_size, image_size)) for img in images]
    return images
    
def save_to_hdf5(hdf5_path, img_path, cls, img, split, compression="gzip", n_trials=10, n_sleep=1):
    for i in range(n_trials):
        try:
            with h5py.File(hdf5_path, "w") as f:
                filename = img_path.split("/")[-1]  
                clsname = filename.split("_")[0]
                save_path = f"{split}/{cls}/{filename}"
                f.create_dataset(save_path, data=img, compression=compression)
                
                f[save_path].attrs["label"] = cls
                f[save_path].attrs["filename"] = filename
                f[save_path].attrs["class"] = clsname
                break
        except BlockingIOError:
            time.sleep(n_sleep)
        
def generate(distributed_state, dataloader, pipe, args, hdf5_path, split, prompt=""):
    for i, batch in tqdm(enumerate(dataloader)):
        images = batch["image"].to(args.device)
        paths = np.array(batch["path"])
        labels = np.array(batch["label"])
        with distributed_state.split_between_processes(list(range(len(images)))) as idxs:
            samples = ddim_sampling(
                pipe=pipe,
                images=images[idxs],
                prompt=prompt,
                num_inference_steps=args.num_steps,
                device=args.device,
                num_samples=args.num_samples,
                ddim_sampling_step=args.ddim_sampling_step,
                guidance_scale=args.guidance_scale,
                do_classifier_free_guidance=True,
                negative_prompt="",
                prefix=f"[{split}] {i}/{len(dataloader)}",
            )  # (B, H, W, C)
        
            samples = postprocess(samples, args.image_size)
            # save random image
            # sample = samples[0]
            # cv2.imwrite(f"debug_{split}.jpg", sample)
            for j, img in enumerate(samples):
                save_to_hdf5(hdf5_path, paths[idxs][j], labels[idxs][j], img, split)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_type", type=str, default="img2img", help="Type of generation: img2img or img2txt")
    parser.add_argument("--inversion_type", type=str, default="ddim", help="Type of inversion: ddim or diffusion")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per real image or caption")
    
    # perturbation parameters
    parser.add_argument("--ddim_perturbation", type=str, default=None, help="Perturbation type for DDIM inversion: None, x0, xt, xT, enc, dec, bottleneck")
    parser.add_argument("--diffusion_perturbation", type=str, default=None, help="Perturbation type for diffusion inversion: None, x0, xt, xT, enc, dec, bottleneck")
    parser.add_argument("--perturbation_scale", type=float, default=1.0, help="Perturbation scale")
    parser.add_argument("--pertubation_dist", type=str, default="uniform", help="Perturbation distribution: uniform or normal")
    
    # diffusion parameters
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Model name for diffusion")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument("--ddim_sampling_step", type=int, default=10, help="Number of reverse steps for DDIM inversion")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for DDIM inversion")
    parser.add_argument("--empty_prompt", action="store_true", help="Use empty prompt for DDIM inversion")
    
    parser.add_argument("--dataset", type=str, default="pets", help="Dataset name")
    parser.add_argument("--data_path", type=str, default="data", help="Directory to save generated samples")
    parser.add_argument("--with_test", action="store_true", help="Generate samples for test set")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--new_dataset_name", type=str, default="new_dataset", help="Name of the new dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated samples")
    
    args = parser.parse_args()
    
    # setup distributed inference
    distributed_state = PartialState()  # this automatically sets up distributed inference
    args.device = distributed_state.device 
    
    # create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if Path(args.output_dir).exists():
        print(f"Output directory {args.output_dir} already exists. It will be overwritten.")
    
    # set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = vars(args)
    print(f"Config: {config}")
    
    dataset_config = {
        "root": args.data_path,
        "download": True,
        "transform": None,
        "train": True,
    }
    if args.dataset == "pets":
        pets_transform = tfms.Compose(
            [
                tfms.transforms.Resize((args.image_size, args.image_size)),
                tfms.transforms.ToTensor(),
            ]
        )
        dataset_config["transform"] = pets_transform
        train_dataset = PetsDataset(**dataset_config)
        dataset_config["train"] = False
        val_dataset = PetsDataset(**dataset_config)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    test_dataloader = DataLoader(
        val_dataset, args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    
    classes = train_dataset.classes
    print(f"Classes: {classes}")
    hdf5_path = os.path.join(args.output_dir, f"{args.new_dataset_name}.hdf5")
    
    pipe = get_diffusion_pipe(args.model_name, args.device)
    if args.inversion_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config
        )
    
    if args.generation_type == "img2img":
        if args.inversion_type == "ddim":
            prompt = "" if args.empty_prompt else NotImplementedError("Prompt is not empty")
            generate(distributed_state, train_dataloader, pipe, args, hdf5_path, split="train", prompt=prompt)

            if args.with_test:
                generate(distributed_state, test_dataloader, pipe, args, hdf5_path, split="test", prompt=prompt)
        else:
            raise NotImplementedError(f"Invalid inversion type: {args.inversion_type}")
    else:
        raise NotImplementedError(f"Invalid generation type: {args.generation_type}")
    
    print(f"Generated samples are saved to {hdf5_path}")
                
                
if __name__ == "__main__":
    main()