import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import json
from model_training_inference import BaseModel, Trainer, InferenceEngine
import pandas as pd


def get_transform_learn(size):
    """Data transformation for training"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_transform_perform(size):
    """Data transformation for inference"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def parse_args():
    """Process command-line arguments"""
    parser = argparse.ArgumentParser(description="Training / Evaluation Pipeline")
    parser.add_argument(
        '--process', type=str, required=True, choices=['train', 'further_train', 'inference'],
        help='Type of process'
    )
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the folder with /train, /validation')
    parser.add_argument('--params_path', type=str, required=True, help='Path to .json with training parameters')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to model weights (if needed)')
    return parser.parse_args()


def main():
    """Main function for training, further training, or inference"""
    args = parse_args()

    with open(args.params_path, 'r') as f:
        params = json.load(f)

    models_list = params["model"] if isinstance(params["model"], list) else [params["model"]]

    for model_name in models_list:
        size = 299 if model_name == "inception_v3" else 224
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = get_transform_learn(size) if args.process != 'inference' else get_transform_perform(size)

        if args.process in ['train', 'further_train']:
            train_path = os.path.join(args.data_folder, "train")
            val_path = os.path.join(args.data_folder, "validation")

            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Folder {train_path} not found")
            if not os.path.exists(val_path):
                raise FileNotFoundError(f"Folder {val_path} not found")

            train_loader = DataLoader(
                datasets.ImageFolder(train_path, transform=transform),
                batch_size=params["batch_size"], shuffle=True
            )

            val_loader = DataLoader(
                datasets.ImageFolder(val_path, transform=transform),
                batch_size=params["batch_size"], shuffle=False
            )

            model = BaseModel(
                model_name=model_name,
                num_classes=params["num_classes"],
                fine_tuning=True,
                device=device
            ).model

            trainer = Trainer(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config_path=args.params_path,
                weights_path=args.weights_path
            )

            trainer.train(params["num_epochs"])

        elif args.process == 'inference':
            inference_engine = InferenceEngine(
                model_name=model_name,
                num_classes=params["num_classes"],
                device=device,
                weights_path=args.weights_path
            )
            inference_engine.predict(args.data_folder)


if __name__ == "__main__":
    main()