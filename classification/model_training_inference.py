import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from PIL import Image
import pandas as pd
from functools import partial


class BaseModel:
    def __init__(self, model_name, num_classes, fine_tuning, device):
        """Инициализация базовой модели."""
        self.model_name = model_name
        self.num_classes = num_classes
        self.fine_tuning = fine_tuning
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        """Загрузка модели и настройка классификатора для fine-tuning."""
        if not self.fine_tuning:
            model = getattr(models, self.model_name)(weights="DEFAULT")
            for param in model.parameters():
                param.requires_grad = False
        else:
            model = getattr(models, self.model_name)(weights=None)
            for param in model.parameters():
                param.requires_grad = True

        # Получаем информацию о классификаторе
        classifier_name, classifier_idx = self.get_classifier_info()
        classifier = getattr(model, classifier_name)
        if classifier_idx is not None:
            classifier = classifier[classifier_idx]

        in_features = classifier.in_features
        new_classifier = nn.Linear(in_features, self.num_classes)

        # Если указан индекс классификатора, меняем его
        if classifier_idx is not None:
            getattr(model, classifier_name)[classifier_idx] = new_classifier
        else:
            setattr(model, classifier_name, new_classifier)

        # Разрешаем обучение параметров нового классификатора
        for param in new_classifier.parameters():
            param.requires_grad = True

        # Дополнительная настройка для InceptionV3 (если используется)
        if "inception_v3" in self.model_name and hasattr(model, "AuxLogits"):
            aux_in_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(aux_in_features, self.num_classes)
            )
            for param in model.AuxLogits.fc.parameters():
                param.requires_grad = True

        return model.to(self.device)

    def get_classifier_info(self):
        """Получение информации о классификаторе для разных типов моделей."""
        classifier_info = {
            "alexnet": ("classifier", -1),
            "vgg": ("classifier", -1),
            "resnet": ("fc", None),
            "resnext": ("fc", None),
            "regnet": ("fc", None),
            "shufflenet": ("fc", None),
            "wide_resnet": ("fc", None),
            "googlenet": ("fc", None),
            "mobilenet": ("classifier", -1),
            "efficientnet": ("classifier", -1),
            "convnext": ("classifier", -1),
            "densenet": ("classifier", None),
            "inception_v3": ("fc", None),
            "squeezenet": ("classifier", 1),
            "swin": ("head", None),
            "maxvit": ("classifier", -1),
            "mnasnet": ("classifier", -1),
            "vit": ("heads", None)
        }

        # Определяем, какой классификатор использовать для модели
        for key, (layer_name, layer_index) in classifier_info.items():
            if key in self.model_name:
                return layer_name, layer_index

        raise ValueError(f"Модель {self.model_name} не поддерживается.")


class Trainer:
    def __init__(self, model_name, model, train_loader, val_loader, config_path, weights_path=None):
        with open(config_path, "r") as f:
            config = json.load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        self.weights_path = weights_path

        self.lr_min = config["lr_min"]
        self.lr_max = config["lr_max"]
        self.learning_rate = config["learning_rate"]
        self.start_epoch = config.get("start_epoch", 0)
        self.num_epochs = config["num_epochs"] + self.start_epoch
        self.batch_size = config["batch_size"]
        self.num_classes = config["num_classes"]
        self.optimizer_name = config["optimizers"]

        total_steps = self.num_epochs * len(self.train_loader)
        self.warmup_steps = int(total_steps * 0.2)

        self.optimizer = self._select_optimizer(model, self.learning_rate, config)

        self.scheduler = LambdaLR(
            self.optimizer,
            partial(self._lr_lambda, warmup_steps=self.warmup_steps, total_steps=total_steps,
                    lr_max=self.lr_max, lr_min=self.lr_min)
        )

        if self.weights_path is None:
            self.weights_dir = os.path.join('./results/training', self.model_name)
        else:
            self.weights_dir = os.path.dirname(self.weights_path)

        os.makedirs(self.weights_dir, exist_ok=True)
        if self.weights_path is None:
            self.weights_path = os.path.join(self.weights_dir, "best_weights.pth")

        if os.path.exists(self.weights_path):
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))

        self.history = []

    def _select_optimizer(self, model, lr, config):
        optimizers = {
            "SGD": optim.SGD,
            "Momentum": lambda params, lr: optim.SGD(params, lr, momentum=config.get("momentum", 0.9)),
            "NAG": lambda params, lr: optim.SGD(params, lr, momentum=config.get("momentum", 0.9), nesterov=True),
            "Adam": lambda params, lr: optim.Adam(params, lr, betas=config.get("betas", (0.9, 0.999)), eps=config.get("eps", 1e-8)),
            "AdamW": lambda params, lr: optim.AdamW(params, lr, betas=config.get("betas", (0.9, 0.999)), weight_decay=config.get("weight_decay", 1e-2)),
            "RMSprop": lambda params, lr: optim.RMSprop(params, lr, alpha=config.get("alpha", 0.99)),
            "Adagrad": optim.Adagrad,
            "Adadelta": optim.Adadelta
        }

        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer {self.optimizer_name}. Available: {list(optimizers.keys())}")

        return optimizers[self.optimizer_name](model.parameters(), lr)

    def _lr_lambda(self, step, warmup_steps, total_steps, lr_max, lr_min):
        if step < warmup_steps:
            return lr_min + (lr_max - lr_min) * (step / warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

    def train(self, additional_epochs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        losses_path = os.path.join(self.weights_dir, "all_losses.csv")

        if os.path.exists(losses_path):
            df_prev = pd.read_csv(losses_path)
            start_epoch = int(df_prev["epoch"].max()) + 1
            self.history = df_prev.to_dict("records")
        else:
            start_epoch = 0
            self.history = []

        best_weights_path = os.path.join(self.weights_dir, "best_weights.pth")
        if os.path.exists(best_weights_path):
            self.model.load_state_dict(torch.load(best_weights_path, map_location=self.device))

        epoch_times = []
        best_acc = 0.0
        best_metrics = {}

        total_epochs = start_epoch + additional_epochs
        pbar = tqdm(range(start_epoch, total_epochs), desc=f"Training model {self.model_name}")

        for epoch in pbar:
            start_time = time.time()
            self.model.train()
            running_loss = 0.0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            avg_loss = round(running_loss / len(self.train_loader), 3)
            val_metrics = self.evaluate()
            self.scheduler.step()

            pbar.set_postfix({
                "Epoch": f"{epoch+1}",
                "Loss": f"{avg_loss:.4f}",
                "Accuracy": f"{val_metrics['accuracy']:.2f}%"
            })

            self.history.append({
                "epoch": epoch,
                "loss": avg_loss
            })

            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
                torch.save(self.model.state_dict(), best_weights_path)

            epoch_times.append(time.time() - start_time)

        avg_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
        self.save_metrics(avg_time, best_acc, best_metrics)

    def evaluate(self):
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return self.compute_metrics(all_labels, all_preds)

    def compute_metrics(self, true_labels, predicted_labels):
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        return {
            "accuracy": round(accuracy, 3),
            "precision_macro": round(precision_score(true_labels, predicted_labels, average="macro", zero_division=0), 3),
            "recall_macro": round(recall_score(true_labels, predicted_labels, average="macro", zero_division=0), 3),
            "f1_macro": round(f1_score(true_labels, predicted_labels, average="macro", zero_division=0), 3),
            "precision_micro": round(precision_score(true_labels, predicted_labels, average="micro", zero_division=0), 3),
            "recall_micro": round(recall_score(true_labels, predicted_labels, average="micro", zero_division=0), 3),
            "f1_micro": round(f1_score(true_labels, predicted_labels, average="micro", zero_division=0), 3)
        }

    def save_metrics(self, avg_time, best_acc, best_metrics):
        os.makedirs(self.weights_dir, exist_ok=True)

        df_losses = pd.DataFrame(self.history)
        df_losses["loss"] = df_losses["loss"].round(3)
        df_losses.drop_duplicates(subset="epoch", keep="last", inplace=True)
        df_losses.sort_values("epoch", inplace=True)

        df_losses.to_csv(os.path.join(self.weights_dir, "all_losses.csv"), index=False)
        df_losses.to_excel(os.path.join(self.weights_dir, "all_losses.xlsx"), index=False)

        with open(os.path.join(self.weights_dir, "best_metrics.txt"), "w") as f:
            f.write(f"Best Accuracy: {best_acc:.2f}%\n")
            for k, v in best_metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write(f"Average Epoch Time (sec): {avg_time:.2f}\n")

        summary_data = {
            "model": self.model_name,
            "avg_epoch_time_sec": avg_time,
            "best_val_accuracy": best_acc,
            **{f"best_val_{k}": v for k, v in best_metrics.items()}
        }

        df_summary = pd.DataFrame([summary_data])
        csv_path = "./results/training/all_times.csv"
        xlsx_path = "./results/training/all_times.xlsx"

        if os.path.exists(csv_path):
            df_all = pd.read_csv(csv_path)
            df_all = df_all[df_all["model"] != self.model_name]
            df_all = pd.concat([df_all, df_summary], ignore_index=True)
        else:
            df_all = df_summary

        df_all.to_csv(csv_path, index=False)
        df_all.to_excel(xlsx_path, index=False)


class InferenceEngine:
    def __init__(self, model_name, num_classes, device, weights_path=None):
        self.device = device
        self.model_name = model_name

        if weights_path is None:
            weights_path = os.path.join("results/training", model_name, "best_weights.pth")

        try:
            self.model = BaseModel(model_name, num_classes, fine_tuning=False, device=device).model
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model from {weights_path}: {e}")
            raise

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        """Prediction for a single image"""
        try:
            image = self.transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None  # Skip corrupted images

        with torch.no_grad():
            output = self.model(image)
            pred = torch.argmax(output, dim=1).item()
        return pred

    def predict(self, input_path):
        """Prediction for a set of images (folder or file)"""
        results = []

        if os.path.isdir(input_path):
            image_files = [os.path.join(root, f) for root, _, files in os.walk(input_path)
                           for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))]

            if not image_files:
                print(f"No images found in folder {input_path} for inference.")
                return

            print(f"Found {len(image_files)} images for inference.")

            image_files.sort()
            for image_path in tqdm(image_files, desc="Inference", ncols=100):  # Progress bar
                pred = self.predict_image(image_path)
                if pred is not None:
                    results.append((os.path.relpath(image_path, input_path), pred))
        elif os.path.isfile(input_path):
            # If it is a file, predict only for one image
            pred = self.predict_image(input_path)
            if pred is not None:
                results.append((os.path.basename(input_path), pred))
        else:
            raise ValueError("Invalid path. Please provide an image file or folder.")

        output_csv = os.path.join("./results", "predictions", f"{self.model_name}.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        df = pd.DataFrame(results, columns=["Filename", "Prediction"])
        df.to_csv(output_csv, index=False)
        print(f"Prediction results saved to {output_csv}")