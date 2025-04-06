import argparse
import sys
from pathlib import Path
from preprocessor import FileProcessor, FolderProcessor

def main():
    parser = argparse.ArgumentParser(description="Image preprocessing")
    parser.add_argument("--input", required=True, help="Path to the folder or file with images")
    parser.add_argument("--output", required=True, help="Path to the output folder")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    log_file = output_path / "error.log"

    def log_error(msg):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    open(log_file, "w").close()

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Path not found: {input_path}")

        if input_path.is_file():
            FileProcessor(output_path, log_error).process(str(input_path))
        elif input_path.is_dir():
            FolderProcessor(output_path, log_error).process(str(input_path))
        else:
            raise ValueError("Unsupported input format!")

        print(f"""
Processing completed!
Results folder: {output_path}
Error log: {log_file}
""")
    except Exception as e:
        log_error(str(e))
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()