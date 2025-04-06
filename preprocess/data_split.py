import argparse
import random
from pathlib import Path
import shutil
from tqdm import tqdm


def split_dataset(input_dir, output_dir, train_ratio=0.8):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    train_path = output_dir / "train"
    val_path = output_dir / "validation"

    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)

    class_folders = [f for f in input_dir.iterdir() if f.is_dir()]

    for class_folder in class_folders:
        images = list(class_folder.glob("*.jpg")) + \
                list(class_folder.glob("*.jpeg")) + \
                list(class_folder.glob("*.png"))

        if not images:
            print(f"Warning: Folder is empty: {class_folder.name}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        train_class_dir = train_path / class_folder.name
        val_class_dir = val_path / class_folder.name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for img in tqdm(train_images, desc=f"Copying to train → {class_folder.name}", leave=False):
            shutil.copy(img, train_class_dir / img.name)

        for img in tqdm(val_images, desc=f"Copying to validation → {class_folder.name}", leave=False):
            shutil.copy(img, val_class_dir / img.name)

    print(f"Dataset splitting completed!\n"
          f"Train folder: {train_path}\n"
          f"Validation folder: {val_path}")


def main():
    parser = argparse.ArgumentParser(description="Image preprocessing and splitting into train/val directories.")
    parser.add_argument("--input", required=True, help="Path to the input directory containing class folders of images.")
    parser.add_argument("--output", required=True, help="Path to the output directory where train/validation structure will be saved.")
    args = parser.parse_args()

    split_dataset(args.input, args.output)


if __name__ == "__main__":
    main()