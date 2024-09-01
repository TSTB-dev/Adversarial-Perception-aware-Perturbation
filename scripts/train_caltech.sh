    # parser.add_argument("--model_name", type=str, default="resnet50")
    # parser.add_argument("--pretrained", action="store_true", default=False)
    # parser.add_argument("--dataset", type=str, default="pets")
    # parser.add_argument("--data_path", type=str, default="./data")
    # parser.add_argument("--batch_size", type=int, default=256)
    # parser.add_argument("--image_size", type=int, default=256)
    # parser.add_argument("--image_channel", type=int, default=3)
    # parser.add_argument("--epochs", type=int, default=100)
    # parser.add_argument("--optimzier", type=str, default="adam")
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--weight_decay", type=float, default=1e-4)
    # parser.add_argument("--scheduler", type=str, default="cosine")
    # parser.add_argument("--scheduler_epochs", type=int, default=100)
    # parser.add_argument("--wandb_project", type=str, default="apap")
    # parser.add_argument("--wandb_run_name", type=str, default="apap-pets-real-resnet50")
    # parser.add_argument("--save_model", action="store_true", default=False)
    # parser.add_argument("--save_freq", type=int, default=None)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--eval_only", action="store_true", default=False)
    # parser.add_argument("--model_path", type=str, default=None)
    # parser.add_argument("--save_best_model", action="store_true", default=False)

export CUDA_VISIBLE_DEVICES=3
python src/train_classifier.py \
    --model_name resnet18 \
    --model_path /home/haselab/projects/sakai/Adversarial-Perception-aware-Perturbation/wandb/run-20240826_203543-c3wjj7dc/files/models/best_classifier_500.pth \
    --eval_only \
    --dataset caltech \
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
    --wandb_project debug \
    --wandb_run_name caltech-synx2-resnet18 \
