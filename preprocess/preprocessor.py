import cv2
import dlib
import numpy as np
from pathlib import Path
from mtcnn import MTCNN
import logging
import os
import sys
import warnings
from contextlib import contextmanager
import tensorflow as tf
from tqdm import tqdm

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

class BaseProcessor:
    def __init__(self, output_path, log_error=None):
        self.output_path = Path(output_path)
        self.log_error = log_error
        
        with suppress_output():
            self.detector = MTCNN(steps_threshold=[0.5, 0.7, 0.9])
            
        model_path = Path(__file__).resolve().parent.parent / "settings" / "face_landmarks" / "shape_predictor_68_face_landmarks.dat"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        with suppress_output():
            self.predictor = dlib.shape_predictor(str(model_path))
            
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _preprocess_image(self, image):
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
        return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def _align_face(self, image, shape):
        left_eye = np.mean([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)], axis=0)
        right_eye = np.mean([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)], axis=0)
        angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
        center = tuple(np.mean([left_eye, right_eye], axis=0))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

class FileProcessor(BaseProcessor):
    def process(self, image_path):
        image_path = Path(image_path)
        with suppress_output():
            image = cv2.imread(str(image_path))
        if image is None:
            if self.log_error:
                self.log_error(f"File not loaded: {image_path}")
            return None
        return self._process_image(image, image_path.name)

    def _process_image(self, image, filename):
        try:
            processed = self._preprocess_image(image)
            with suppress_output():
                faces = self.detector.detect_faces(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            
            if not faces:
                if self.log_error:
                    self.log_error(f"Face not detected in: {filename}")
                return None

            main_face = max(faces, key=lambda x: x['confidence'])
            x, y, w, h = map(int, main_face['box'])
            x, y = max(0, x), max(0, y)

            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            rect = dlib.rectangle(x, y, x + w, y + h)
            with suppress_output():
                shape = self.predictor(gray, rect)

            aligned = self._align_face(processed, shape)
            cropped = self._crop_face(aligned)

            output_file = self.output_path / filename
            cv2.imwrite(str(output_file), cropped)
            return output_file
        except Exception as e:
            if self.log_error:
                self.log_error(f"Error processing {filename}: {str(e)}")
            return None

    def _crop_face(self, image):
        with suppress_output():
            faces = self.detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if faces:
            x, y, w, h = faces[0]['box']
            return cv2.resize(image[y:y + h, x:x + w], (224, 224))
        return cv2.resize(image, (224, 224))

class FolderProcessor(BaseProcessor):
    def process(self, folder_path):
        folder = Path(folder_path)
        valid_extensions = {'.jpg', '.jpeg', '.png'}

        all_files = [
            f for f in folder.rglob("*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        # Прогресс-бар с принудительным выводом в stdout
        with tqdm(total=len(all_files), 
                 desc="Processing images", 
                 unit="img",
                 dynamic_ncols=True,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Elapsed: {elapsed}]",
                 file=sys.stdout) as pbar:
            
            for file in all_files:
                try:
                    relative = file.relative_to(folder_path)
                    subfolder = self.output_path / relative.parent
                    subfolder.mkdir(parents=True, exist_ok=True)
                    with suppress_output():
                        FileProcessor(subfolder, self.log_error).process(file)
                except Exception as e:
                    if self.log_error:
                        self.log_error(f"Error processing {file}: {str(e)}")
                finally:
                    pbar.update(1)


