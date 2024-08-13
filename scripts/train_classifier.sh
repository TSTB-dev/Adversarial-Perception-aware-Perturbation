
python src/train_classifier.py \
    --model_name resnet50 \
    --pretrained \
    --dataset pets \
    --data_path ./data \
    --batch_size 256 \
    --image_size 256 \
    --image_channel 3 \
    --epochs 50 \
    --optimzier adam \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --scheduler cosine \
    --scheduler_epochs 50 \
    --wandb_project apap \
    --wandb_run_name apap-real-resnet50 \
    --save_model \
