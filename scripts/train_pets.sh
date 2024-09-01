

export CUDA_VISIBLE_DEVICES=1
python src/train_classifier.py \
    --model_name resnet18 \
    --syn_data_path /home/haselab/projects/sakai/Adversarial-Perception-aware-Perturbation/syn_data/unclip_pets_unclip_s10_n20_x2.hdf5 \
    --syn_dataset \
    --dataset pets \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 500 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name pets-synx2-resnet18 \

python src/train_classifier.py \
    --model_name resnet18 \
    --syn_data_path /home/haselab/projects/sakai/Adversarial-Perception-aware-Perturbation/syn_data/unclip_pets_unclip_s10_n20_x4.hdf5 \
    --syn_dataset \
    --dataset pets \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 500 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name pets-synx4-resnet18 \

python src/train_classifier.py \
    --model_name resnet18 \
    --syn_data_path /home/haselab/projects/sakai/Adversarial-Perception-aware-Perturbation/syn_data/unclip_pets_unclip_s10_n20_x8.hdf5 \
    --syn_dataset \
    --dataset pets \
    --data_path ./data \
    --batch_size 64 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 500 \
    --optimzier adam \
    --lr 5e-4 \
    --weight_decay 0 \
    --save_model \
    --save_freq 500 \
    --save_best_model \
    --scheduler cosine \
    --scheduler_epochs 200 \
    --wandb_project unclip \
    --wandb_run_name pets-synx8-resnet18 \
    