accelerate launch src/generate_dataset.py \
    --generation_type img2img \
    --inversion_type ddim \
    --num_samples 1 \
    --model_name runwayml/stable-diffusion-v1-5 \
    --num_steps 50 \
    --ddim_sampling_step 20 \
    --guidance_scale 1.0 \
    --empty_prompt \
    --dataset pets \
    --data_path data \
    --with_test \
    --image_size 256 \
    --device cuda \
    --output_dir output \
    --batch_size 256 \
    --new_dataset_name pets_ddim_n50_p20_s3.5 \

    # parser.add_argument("--generation_type", type=str, default="img2img", help="Type of generation: img2img or img2txt")
    # parser.add_argument("--inversion_type", type=str, default="ddim", help="Type of inversion: ddim or diffusion")
    # parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per real image or caption")
    
    # # perturbation parameters
    # parser.add_argument("--ddim_perturbation", type=str, default=None, help="Perturbation type for DDIM inversion: None, x0, xt, xT, enc, dec, bottleneck")
    # parser.add_argument("--diffusion_perturbation", type=str, default=None, help="Perturbation type for diffusion inversion: None, x0, xt, xT, enc, dec, bottleneck")
    # parser.add_argument("--perturbation_scale", type=float, default=1.0, help="Perturbation scale")
    # parser.add_argument("--pertubation_dist", type=str, default="uniform", help="Perturbation distribution: uniform or normal")
    
    # # diffusion parameters
    # parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Model name for diffusion")
    # parser.add_argument("--num_steps", type=int, default=100, help="Number of diffusion steps")
    # parser.add_argument("--ddim_sampling_step", type=int, default=10, help="Number of reverse steps for DDIM inversion")
    # parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale for DDIM inversion")
    # parser.add_argument("--empty_prompt", action="store_true", help="Use empty prompt for DDIM inversion")
    
    # parser.add_argument("--data_path", type=str, default="data", help="Directory to save generated samples")
    # parser.add_argument("--with_test", action="store_true", help="Generate samples for test set")
    # parser.add_argument("--image_size", type=int, default=256, help="Image size")
    # parser.add_argument("--device", type=str, default="cuda", help="Device to run the model")
    # parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # parser.add_argument("--output_dir", type=str, default="output", help="Directory to save generated samples")