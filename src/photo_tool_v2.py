from __future__ import annotations

import argparse
import csv
import io
import math
import shutil
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import cv2
except ImportError:  # pragma: no cover - runtime dependency
    cv2 = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime dependency
    mp = None

try:
    import rawpy
except ImportError:  # pragma: no cover - runtime dependency
    rawpy = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - runtime dependency
    YOLO = None


STANDARD_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_IMAGE_EXTENSIONS = {
    ".rw2",  # Panasonic
    ".arw",  # Sony
    ".cr2",  # Canon
    ".cr3",  # Canon
    ".nef",  # Nikon
    ".orf",  # Olympus
    ".raf",  # Fuji
    ".dng",  # DNG
}
IMAGE_EXTENSIONS = STANDARD_IMAGE_EXTENSIONS | RAW_IMAGE_EXTENSIONS
DECISION_NAMES = ("reject", "review", "keep", "best")
OUTPUT_MODE_CLASSIFIED = "classified"
OUTPUT_MODE_EFFECTS = "effects"
OUTPUT_MODE_BOTH = "both"
OUTPUT_MODE_CHOICES = (
    OUTPUT_MODE_CLASSIFIED,
    OUTPUT_MODE_EFFECTS,
    OUTPUT_MODE_BOTH,
)
RECOGNITION_EFFECTS_DIRNAME = "_recognition_effects"
LEFT_EYE_INDEXES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDEXES = (362, 385, 387, 263, 373, 380)
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_landmarker.task"
DEFAULT_INPUT_PATH = Path(__file__).resolve().parent.parent / "training_data" / "data4"
DEFAULT_YOLO_MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolov8n-face.pt"
PRIMARY_FACE_AREA_TOLERANCE = 0.85
YUNET_GROUP_MIN_FACE_COUNT = 3
YUNET_GROUP_MIN_FACE_RATIO = 0.0005
YUNET_PAIR_MIN_FACE_COUNT = 2
YUNET_PAIR_MIN_FACE_RATIO = 0.004
YUNET_VALIDATED_PERSON_MIN_FACE_RATIO = 0.001
YUNET_VALIDATED_PERSON_MIN_FACE_PIXELS = 40
YUNET_UNVALIDATED_PERSON_MIN_FACE_RATIO = 0.006
YUNET_UNVALIDATED_PERSON_MIN_FACE_PIXELS = 90
HAAR_VALIDATED_PERSON_MIN_FACE_RATIO = 0.008
HAAR_VALIDATED_PERSON_MIN_FACE_PIXELS = 120
HAAR_VALIDATED_PERSON_MAX_CENTER_DISTANCE = 0.55
CLOSED_EYE_PERIPHERAL_CENTER_DISTANCE = 0.6
CLOSED_EYE_SIMILAR_FACE_RATIO = 0.85
CENTER_PRIORITY_FACE_COUNT = 3
YUNET_PRIORITY_FACE_COUNT = 2
LANDMARK_BRIGHTEN_FACTOR = 1.35
YUNET_VALIDATED_PERSON_MIN_SCORE = 0.72
YUNET_UNVALIDATED_PERSON_MIN_SCORE = 0.72
ROTATED_YUNET_SCORE_THRESHOLD = 0.7
ROTATED_YUNET_MIN_AREA_RATIO = 0.01
ROTATED_YUNET_MAX_ASPECT_RATIO = 1.8
EYES_OPEN_WEIGHT = 0.45
BLUR_WEIGHT = 0.30
EXPOSURE_WEIGHT = 0.12
FACE_SIZE_WEIGHT = 0.03
PERSON_WEIGHT = 0.10
PRIMARY_FACE_CLOSED_PENALTY = 0.55
SECONDARY_CLOSED_FACE_PENALTY = 0.10
SECONDARY_CLOSED_FACE_PENALTY_CAP = 0.30
NON_PERSON_PENALTY = 0.22
GROUP_FACE_PENALTY = 0.03
PRIMARY_FACE_CLOSED_REVIEW_THRESHOLD = 0.42


@dataclass
class FaceResult:
    face_id: int
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    is_closed: bool
    bbox: tuple[int, int, int, int]
    area_ratio: float
    center_distance_ratio: float
    source: str
    score: float | None
    is_validated: bool
    is_primary: bool = False
    is_center_priority: bool = False


@dataclass
class PhotoAssessment:
    image_path: Path
    decision: str
    scene_category: str
    is_person: bool
    usable_score: float
    reasons: list[str]
    legacy_status: str
    primary_face_bbox: tuple[int, int, int, int] | None
    face_count: int
    closed_face_count: int
    min_ear: float | None
    blur_score: float
    exposure_score: float
    eyes_open_prob: float
    faces: list[FaceResult]
    primary_face_id: int | None
    primary_face_ear: float | None
    processing_seconds: float = 0.0


@dataclass
class FaceCandidate:
    bbox: tuple[int, int, int, int]
    area_ratio: float
    center_distance_ratio: float
    score: float | None
    source: str


class FaceLandmarkDetector:
    def detect(self, rgb_image: np.ndarray):
        raise NotImplementedError

    def close(self) -> None:
        return None


class TaskFaceLandmarkerDetector(FaceLandmarkDetector):
    def __init__(self, model_path: Path, max_faces: int) -> None:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self._landmarker.detect(mp_image)
        return result.face_landmarks or []

    def close(self) -> None:
        self._landmarker.close()


class FaceCandidateDetector:
    def detect(self, rgb_image: np.ndarray) -> list[FaceCandidate]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class YoloFaceDetector(FaceCandidateDetector):
    def __init__(
        self,
        model_path: Path,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
        min_face_pixels: int,
        max_candidates: int,
    ) -> None:
        if YOLO is None:
            raise SystemExit(
                "Missing dependency: ultralytics\n"
                "Install it first:\n"
                "  python -m pip install ultralytics"
            )
        self._model = YOLO(str(model_path))
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = max(1, top_k)
        self.min_face_pixels = min_face_pixels
        self.max_candidates = max_candidates
        self.face_class_ids = self._resolve_face_class_ids()

    def _resolve_face_class_ids(self) -> set[int] | None:
        names = getattr(self._model, "names", None)
        if names is None:
            return None

        if isinstance(names, dict):
            class_map = {int(key): str(value) for key, value in names.items()}
        elif isinstance(names, (list, tuple)):
            class_map = {index: str(value) for index, value in enumerate(names)}
        else:
            return None

        face_class_ids = {
            class_id
            for class_id, class_name in class_map.items()
            if "face" in class_name.lower()
        }
        if face_class_ids:
            return face_class_ids
        if len(class_map) == 1:
            return set(class_map.keys())
        return None

    def detect(self, rgb_image: np.ndarray) -> list[FaceCandidate]:
        candidates = self._detect_yolo(rgb_image, source="yolo")
        if not candidates:
            candidates = self._detect_rotated_yolo(rgb_image)
        if not candidates:
            return []
        return non_max_suppression(
            candidates,
            max_candidates=self.max_candidates,
            iou_threshold=0.35,
        )

    def _predict(self, rgb_image: np.ndarray):
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        prediction = self._model.predict(
            source=bgr_image,
            conf=self.score_threshold,
            iou=self.nms_threshold,
            max_det=self.top_k,
            verbose=False,
        )
        if not prediction:
            return None
        return prediction[0]

    def _detect_yolo(self, rgb_image: np.ndarray, source: str) -> list[FaceCandidate]:
        height, width = rgb_image.shape[:2]
        prediction = self._predict(rgb_image)
        if prediction is None or prediction.boxes is None:
            return []

        candidates: list[FaceCandidate] = []
        for box in prediction.boxes:
            class_id = int(box.cls[0]) if box.cls is not None else 0
            if self.face_class_ids is not None and class_id not in self.face_class_ids:
                continue

            confidence = float(box.conf[0]) if box.conf is not None else None
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            clipped_bbox = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(width - 1, int(round(x2))),
                min(height - 1, int(round(y2))),
            )
            box_width = clipped_bbox[2] - clipped_bbox[0]
            box_height = clipped_bbox[3] - clipped_bbox[1]
            if box_width < self.min_face_pixels or box_height < self.min_face_pixels:
                continue
            candidates.append(
                create_face_candidate(
                    bbox=clipped_bbox,
                    image_width=width,
                    image_height=height,
                    score=confidence,
                    source=source,
                )
            )
        return candidates

    def _detect_rotated_yolo(self, rgb_image: np.ndarray) -> list[FaceCandidate]:
        height, width = rgb_image.shape[:2]
        rotated_modes = (
            ("rot90_ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ("rot90_cw", cv2.ROTATE_90_CLOCKWISE),
        )
        rotated_candidates: list[FaceCandidate] = []
        for rotation_name, rotation_code in rotated_modes:
            rotated_image = cv2.rotate(rgb_image, rotation_code)
            candidates = self._detect_yolo(
                rotated_image,
                source=f"yolo_{rotation_name}",
            )
            for candidate in candidates:
                if candidate.score is None or candidate.score < ROTATED_YUNET_SCORE_THRESHOLD:
                    continue
                mapped_bbox = map_bbox_from_rotated_to_original_image(
                    bbox=candidate.bbox,
                    rotation_name=rotation_name,
                    image_width=width,
                    image_height=height,
                )
                mapped_candidate = create_face_candidate(
                    bbox=mapped_bbox,
                    image_width=width,
                    image_height=height,
                    score=candidate.score,
                    source=f"yolo_{rotation_name}",
                )
                if mapped_candidate.area_ratio < ROTATED_YUNET_MIN_AREA_RATIO:
                    continue
                mapped_width = mapped_bbox[2] - mapped_bbox[0]
                mapped_height = mapped_bbox[3] - mapped_bbox[1]
                min_dimension = max(1, min(mapped_width, mapped_height))
                max_dimension = max(mapped_width, mapped_height)
                aspect_ratio = max_dimension / min_dimension
                if aspect_ratio > ROTATED_YUNET_MAX_ASPECT_RATIO:
                    continue
                rotated_candidates.append(mapped_candidate)

        return rotated_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect suspected closed eyes in portrait image and RAW files."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to an image file or a folder containing supported image/RAW files. "
            f"Defaults to: {DEFAULT_INPUT_PATH}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eye_review_output"),
        help="Directory for annotated images and CSV output.",
    )
    parser.add_argument(
        "--output-mode",
        choices=OUTPUT_MODE_CHOICES,
        default=OUTPUT_MODE_CLASSIFIED,
        help=(
            "Output selection: classified=only copy originals into reject/review/keep/best, "
            "effects=only save annotated recognition previews, both=save both."
        ),
    )
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.18,
        help="EAR threshold below which a face is marked as closed-eye.",
    )
    parser.add_argument(
        "--min-face-size",
        type=float,
        default=0.0025,
        help="Only allow closed-eye decisions for faces above this image-area ratio.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=12,
        help="Maximum number of validated faces to keep per image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MediaPipe face landmarker .task model file.",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Compatibility alias for --output-mode both.",
    )
    parser.add_argument(
        "--yolo-model-path",
        type=Path,
        default=DEFAULT_YOLO_MODEL_PATH,
        help="Path to the YOLOv8 face detection model (.pt/.onnx).",
    )
    parser.add_argument(
        "--yolo-score-threshold",
        type=float,
        default=0.6,
        help="Minimum score for YOLOv8 face detections.",
    )
    parser.add_argument(
        "--yolo-nms-threshold",
        type=float,
        default=0.3,
        help="NMS threshold for YOLOv8 face detections.",
    )
    parser.add_argument(
        "--yolo-top-k",
        type=int,
        default=5000,
        help="Top-K candidate limit for YOLOv8 before NMS.",
    )
    parser.add_argument(
        "--min-face-pixels",
        type=int,
        default=20,
        help="Minimum candidate face size in pixels for detector filtering.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=20,
        help="Maximum number of face candidates to validate per image.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=2.5,
        help="How much to expand a detected face before landmark validation.",
    )
    parser.add_argument(
        "--landmark-min-crop-size",
        type=int,
        default=256,
        help="Upscale small face crops to at least this size before landmarks.",
    )
    args = parser.parse_args()
    if args.save_annotated and args.output_mode == OUTPUT_MODE_CLASSIFIED:
        args.output_mode = OUTPUT_MODE_BOTH
    return args


def ensure_dependencies() -> None:
    if mp is None:
        raise SystemExit(
            "Missing dependency: mediapipe\n"
            "Install it first:\n"
            "  python -m pip install mediapipe pillow numpy rawpy"
        )
    if cv2 is None:
        raise SystemExit(
            "Missing dependency: opencv-python\n"
            "Install it first:\n"
            "  python -m pip install opencv-python"
        )
    if YOLO is None:
        raise SystemExit(
            "Missing dependency: ultralytics\n"
            "Install it first:\n"
            "  python -m pip install ultralytics"
        )
    if not has_task_mediapipe_api():
        raise SystemExit(
            "Unsupported mediapipe installation. Expected the mp.tasks vision API."
        )


def has_task_mediapipe_api() -> bool:
    return hasattr(mp, "tasks") and hasattr(mp.tasks, "vision")


def iter_images(input_path: Path, exclude_dirs: list[Path] | None = None) -> list[Path]:
    exclude_dirs = [path.resolve() for path in (exclude_dirs or [])]
    if input_path.is_file():
        return [input_path]
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        and not is_path_in_directories(path.resolve(), exclude_dirs)
    )


def is_path_in_directories(path: Path, directories: list[Path]) -> bool:
    for directory in directories:
        if path == directory or directory in path.parents:
            return True
    return False


def ensure_raw_support(image_paths: list[Path]) -> None:
    requires_rawpy = any(path.suffix.lower() in RAW_IMAGE_EXTENSIONS for path in image_paths)
    if requires_rawpy and rawpy is None:
        raise SystemExit(
            "RAW image support requires rawpy.\n"
            "Install it first:\n"
            "  python -m pip install rawpy"
        )


def ensure_task_model(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading face landmarker model to: {model_path}")
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
    except Exception as exc:  # pragma: no cover - network dependent
        raise SystemExit(
            "Failed to download MediaPipe face landmarker model.\n"
            f"Download it manually from:\n  {DEFAULT_MODEL_URL}\n"
            f"And place it at:\n  {model_path}\n"
            f"Original error: {exc}"
        ) from exc
    return model_path


def ensure_yolo_model(model_path: Path) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.exists():
        return model_path

    raise SystemExit(
        "YOLOv8 face model not found.\n"
        f"Expected path:\n  {model_path}\n"
        "Use --yolo-model-path to provide a local YOLOv8 face model (.pt/.onnx)."
    )


def create_detector(max_faces: int, model_path: Path) -> FaceLandmarkDetector:
    if has_task_mediapipe_api():
        resolved_model_path = ensure_task_model(model_path)
        return TaskFaceLandmarkerDetector(
            model_path=resolved_model_path,
            max_faces=max_faces,
        )

    raise RuntimeError("No supported MediaPipe face landmark detector available.")


def create_face_candidate_detector(args: argparse.Namespace) -> FaceCandidateDetector:
    model_path = ensure_yolo_model(args.yolo_model_path)
    return YoloFaceDetector(
        model_path=model_path,
        score_threshold=args.yolo_score_threshold,
        nms_threshold=args.yolo_nms_threshold,
        top_k=args.yolo_top_k,
        min_face_pixels=args.min_face_pixels,
        max_candidates=args.max_candidates,
    )


def load_standard_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_raw_preview_image(image_path: Path) -> Image.Image:
    assert rawpy is not None

    with rawpy.imread(str(image_path)) as raw:
        try:
            thumb = raw.extract_thumb()
            if thumb.format == rawpy.ThumbFormat.JPEG:
                with Image.open(io.BytesIO(thumb.data)) as thumb_image:
                    return ImageOps.exif_transpose(thumb_image).convert("RGB")
            if thumb.format == rawpy.ThumbFormat.BITMAP:
                return Image.fromarray(thumb.data).convert("RGB")
        except rawpy.LibRawError:
            pass

        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=True,
            no_auto_bright=False,
            output_bps=8,
        )
        return Image.fromarray(rgb).convert("RGB")


def load_review_image(image_path: Path) -> Image.Image:
    if image_path.suffix.lower() in RAW_IMAGE_EXTENSIONS:
        return load_raw_preview_image(image_path)
    return load_standard_image(image_path)


def load_rgb_image(image_path: Path) -> np.ndarray:
    return np.array(load_review_image(image_path))


def landmark_to_xy(landmark, width: int, height: int) -> tuple[float, float]:
    return landmark.x * width, landmark.y * height


def distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.dist(point_a, point_b)


def compute_ear(
    landmarks, indexes: tuple[int, int, int, int, int, int], width: int, height: int
) -> float:
    points = [landmark_to_xy(landmarks[index], width, height) for index in indexes]
    horizontal = distance(points[0], points[3])
    if horizontal == 0:
        return 0.0
    vertical_a = distance(points[1], points[5])
    vertical_b = distance(points[2], points[4])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def compute_bbox(landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [landmark.x * width for landmark in landmarks]
    ys = [landmark.y * height for landmark in landmarks]
    x1 = max(0, int(min(xs)))
    y1 = max(0, int(min(ys)))
    x2 = min(width - 1, int(max(xs)))
    y2 = min(height - 1, int(max(ys)))
    return x1, y1, x2, y2


def compute_area_ratio(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> float:
    x1, y1, x2, y2 = bbox
    bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
    image_area = image_width * image_height
    if image_area == 0:
        return 0.0
    return bbox_area / image_area


def compute_center_distance_ratio(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> float:
    x1, y1, x2, y2 = bbox
    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0
    max_distance = math.dist((0.0, 0.0), (image_center_x, image_center_y))
    if max_distance == 0:
        return 0.0
    return math.dist((face_center_x, face_center_y), (image_center_x, image_center_y)) / max_distance


def create_face_candidate(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    score: float | None,
    source: str,
) -> FaceCandidate:
    return FaceCandidate(
        bbox=bbox,
        area_ratio=compute_area_ratio(bbox, image_width, image_height),
        center_distance_ratio=compute_center_distance_ratio(
            bbox, image_width, image_height
        ),
        score=score,
        source=source,
    )


def create_face_result_from_candidate(
    candidate: FaceCandidate,
    face_id: int,
) -> FaceResult:
    return FaceResult(
        face_id=face_id,
        ear=None,
        left_ear=None,
        right_ear=None,
        is_closed=False,
        bbox=candidate.bbox,
        area_ratio=candidate.area_ratio,
        center_distance_ratio=candidate.center_distance_ratio,
        source=candidate.source,
        score=candidate.score,
        is_validated=False,
    )


def create_face_result_from_landmarks(
    landmarks,
    face_id: int,
    image_width: int,
    image_height: int,
    ear_threshold: float,
    source: str,
    score: float | None = None,
) -> FaceResult:
    bbox = compute_bbox(landmarks, image_width, image_height)
    left_ear = compute_ear(landmarks, LEFT_EYE_INDEXES, image_width, image_height)
    right_ear = compute_ear(landmarks, RIGHT_EYE_INDEXES, image_width, image_height)
    avg_ear = (left_ear + right_ear) / 2.0
    return FaceResult(
        face_id=face_id,
        ear=avg_ear,
        left_ear=left_ear,
        right_ear=right_ear,
        is_closed=avg_ear < ear_threshold,
        bbox=bbox,
        area_ratio=compute_area_ratio(bbox, image_width, image_height),
        center_distance_ratio=compute_center_distance_ratio(
            bbox, image_width, image_height
        ),
        source=source,
        score=score,
        is_validated=True,
    )


def should_try_full_image_landmarks(candidate: FaceCandidate) -> bool:
    box_width = candidate.bbox[2] - candidate.bbox[0]
    box_height = candidate.bbox[3] - candidate.bbox[1]
    return candidate.area_ratio >= 0.04 or min(box_width, box_height) >= 280


def compute_iou(
    bbox_a: tuple[int, int, int, int], bbox_b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    candidates: list[FaceCandidate],
    max_candidates: int,
    iou_threshold: float,
) -> list[FaceCandidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.score is not None,
            candidate.score if candidate.score is not None else -1.0,
            candidate.area_ratio,
            -candidate.center_distance_ratio,
        ),
        reverse=True,
    )
    kept_candidates: list[FaceCandidate] = []
    for candidate in sorted_candidates:
        if any(
            compute_iou(candidate.bbox, kept_candidate.bbox) >= iou_threshold
            for kept_candidate in kept_candidates
        ):
            continue
        kept_candidates.append(candidate)
        if len(kept_candidates) >= max_candidates:
            break
    return kept_candidates


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    center_x = x1 + (box_width / 2.0)
    center_y = y1 + (box_height / 2.0)
    expanded_width = box_width * scale
    expanded_height = box_height * scale
    crop_x1 = max(0, int(round(center_x - (expanded_width / 2.0))))
    crop_y1 = max(0, int(round(center_y - (expanded_height / 2.0))))
    crop_x2 = min(image_width, int(round(center_x + (expanded_width / 2.0))))
    crop_y2 = min(image_height, int(round(center_y + (expanded_height / 2.0))))
    return crop_x1, crop_y1, crop_x2, crop_y2


def prepare_landmark_crop(
    rgb_image: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    landmark_min_crop_size: int,
) -> tuple[np.ndarray, float]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
    crop_image = rgb_image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_height, crop_width = crop_image.shape[:2]
    if crop_height == 0 or crop_width == 0:
        return crop_image, 1.0

    scale = max(1.0, landmark_min_crop_size / min(crop_width, crop_height))
    if scale <= 1.0:
        return crop_image, 1.0

    resized_crop = cv2.resize(
        crop_image,
        (int(round(crop_width * scale)), int(round(crop_height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized_crop, scale


def generate_landmark_variants(crop_image: np.ndarray) -> list[np.ndarray]:
    variants = [crop_image]

    brightened = np.clip(
        crop_image.astype(np.float32) * LANDMARK_BRIGHTEN_FACTOR,
        0,
        255,
    ).astype(np.uint8)
    variants.append(brightened)

    yuv_image = cv2.cvtColor(crop_image, cv2.COLOR_RGB2YUV)
    yuv_image[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
        yuv_image[:, :, 0]
    )
    clahe_enhanced = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)
    variants.append(clahe_enhanced)
    return variants


def map_bbox_to_original_image(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
    resize_scale: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    crop_x1, crop_y1, _, _ = crop_bbox
    mapped_x1 = crop_x1 + int(round(bbox[0] / resize_scale))
    mapped_y1 = crop_y1 + int(round(bbox[1] / resize_scale))
    mapped_x2 = crop_x1 + int(round(bbox[2] / resize_scale))
    mapped_y2 = crop_y1 + int(round(bbox[3] / resize_scale))
    return (
        max(0, mapped_x1),
        max(0, mapped_y1),
        min(image_width - 1, mapped_x2),
        min(image_height - 1, mapped_y2),
    )


def map_point_from_rotated_to_original_image(
    x: int,
    y: int,
    rotation_name: str,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    if rotation_name == "rot90_ccw":
        mapped_x = image_width - 1 - y
        mapped_y = x
    elif rotation_name == "rot90_cw":
        mapped_x = y
        mapped_y = image_height - 1 - x
    else:
        raise ValueError(f"Unsupported rotation: {rotation_name}")
    return mapped_x, mapped_y


def map_bbox_from_rotated_to_original_image(
    bbox: tuple[int, int, int, int],
    rotation_name: str,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    corners = (
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
    )
    mapped_corners = [
        map_point_from_rotated_to_original_image(
            x=x,
            y=y,
            rotation_name=rotation_name,
            image_width=image_width,
            image_height=image_height,
        )
        for x, y in corners
    ]
    xs = [x for x, _ in mapped_corners]
    ys = [y for _, y in mapped_corners]
    return (
        max(0, min(xs)),
        max(0, min(ys)),
        min(image_width - 1, max(xs)),
        min(image_height - 1, max(ys)),
    )


def select_best_landmarks(
    landmarks_list,
    image_width: int,
    image_height: int,
):
    if not landmarks_list:
        return None
    return max(
        landmarks_list,
        key=lambda landmarks: compute_area_ratio(
            compute_bbox(landmarks, image_width, image_height),
            image_width,
            image_height,
        ),
    )


def select_best_landmarks_for_candidate(
    landmarks_list,
    candidate_bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
):
    if not landmarks_list:
        return None
    return max(
        landmarks_list,
        key=lambda landmarks: compute_iou(
            compute_bbox(landmarks, image_width, image_height),
            candidate_bbox,
        ),
    )


def select_primary_face(faces: list[FaceResult]) -> FaceResult:
    max_area_ratio = max(face.area_ratio for face in faces)
    comparable_faces = [
        face
        for face in faces
        if face.area_ratio >= (max_area_ratio * PRIMARY_FACE_AREA_TOLERANCE)
    ]
    primary_face = min(
        comparable_faces,
        key=lambda face: (face.center_distance_ratio, -face.area_ratio),
    )
    for face in faces:
        face.is_primary = face.face_id == primary_face.face_id
    return primary_face


def select_center_priority_faces(
    faces: list[FaceResult],
    limit: int = CENTER_PRIORITY_FACE_COUNT,
) -> list[FaceResult]:
    return sorted(
        faces,
        key=lambda face: (face.center_distance_ratio, -face.area_ratio),
    )[:limit]


def reindex_faces(faces: list[FaceResult]) -> list[FaceResult]:
    reindexed_faces = sorted(faces, key=lambda face: face.face_id)
    for index, face in enumerate(reindexed_faces, start=1):
        face.face_id = index
    return reindexed_faces


def is_detector_face_source(source: str) -> bool:
    return source.startswith("yolo")


def select_yolo_priority_faces(
    faces: list[FaceResult],
    limit: int = YUNET_PRIORITY_FACE_COUNT,
) -> list[FaceResult]:
    detector_faces = [
        face
        for face in faces
        if is_detector_face_source(face.source)
        and is_plausible_yolo_face(face)
    ]
    return sorted(
        detector_faces,
        key=lambda face: (-face.area_ratio, face.center_distance_ratio),
    )[:limit]


def has_reliable_eye_measurement(face: FaceResult, min_face_size: float) -> bool:
    if face.ear is None or face.left_ear is None or face.right_ear is None:
        return False

    if not face.is_validated:
        return False

    if face.area_ratio < min_face_size:
        return False

    box_width = face.bbox[2] - face.bbox[0]
    box_height = face.bbox[3] - face.bbox[1]
    if min(box_width, box_height) < 55:
        return False

    return True


def has_reliable_open_eye_face(
    faces: list[FaceResult],
    ear_threshold: float,
    min_face_size: float,
) -> bool:
    for face in faces:
        if not has_reliable_eye_measurement(face, min_face_size):
            continue
        if face.ear >= ear_threshold:
            return True
    return False


def is_plausible_yolo_face(face: FaceResult) -> bool:
    box_width = face.bbox[2] - face.bbox[0]
    box_height = face.bbox[3] - face.bbox[1]
    min_dimension = max(1, min(box_width, box_height))
    max_dimension = max(box_width, box_height)
    aspect_ratio = max_dimension / min_dimension
    if aspect_ratio > ROTATED_YUNET_MAX_ASPECT_RATIO:
        return False

    if face.score is None:
        return True

    min_score = (
        YUNET_VALIDATED_PERSON_MIN_SCORE
        if face.is_validated
        else YUNET_UNVALIDATED_PERSON_MIN_SCORE
    )
    return face.score >= min_score


def should_classify_closed_eye_from_center_faces(
    faces: list[FaceResult],
    ear_threshold: float,
    min_face_size: float,
) -> bool:
    center_faces = select_center_priority_faces(faces, limit=CENTER_PRIORITY_FACE_COUNT)
    for face in center_faces:
        if not has_reliable_eye_measurement(face, min_face_size):
            continue
        if abs(face.left_ear - face.right_ear) > 0.12:
            continue
        if face.ear < ear_threshold:
            return True
    return False


def should_classify_closed_eye(
    primary_face: FaceResult,
    faces: list[FaceResult],
    ear_threshold: float,
    min_face_size: float,
) -> bool:
    if len(faces) > 3:
        return False

    if not has_reliable_eye_measurement(primary_face, min_face_size):
        return False

    is_large_single_face = len(faces) == 1 and primary_face.area_ratio >= 0.05
    if (
        is_large_single_face
        and min(primary_face.left_ear, primary_face.right_ear) < 0.09
        and primary_face.ear < 0.16
    ):
        return True

    if abs(primary_face.left_ear - primary_face.right_ear) > 0.12:
        return False

    sorted_area_ratios = sorted((face.area_ratio for face in faces), reverse=True)
    second_face_area_ratio = sorted_area_ratios[1] if len(sorted_area_ratios) > 1 else 0.0
    if (
        len(faces) >= 2
        and primary_face.center_distance_ratio > CLOSED_EYE_PERIPHERAL_CENTER_DISTANCE
        and second_face_area_ratio >= (CLOSED_EYE_SIMILAR_FACE_RATIO * primary_face.area_ratio)
    ):
        return False

    is_dominant_face = (
        len(faces) == 1
        or primary_face.area_ratio >= 0.01
        or (
            second_face_area_ratio > 0
            and primary_face.area_ratio >= (1.8 * second_face_area_ratio)
        )
        or primary_face.center_distance_ratio <= 0.35
    )
    if not is_dominant_face:
        return False

    return primary_face.ear < ear_threshold


def should_classify_as_person(primary_face: FaceResult, faces: list[FaceResult]) -> bool:
    box_width = primary_face.bbox[2] - primary_face.bbox[0]
    box_height = primary_face.bbox[3] - primary_face.bbox[1]
    min_dimension = min(box_width, box_height)
    detector_faces = [
        face
        for face in faces
        if is_detector_face_source(face.source)
        and is_plausible_yolo_face(face)
    ]
    max_detector_face_ratio = max((face.area_ratio for face in detector_faces), default=0.0)

    if (
        len(detector_faces) >= YUNET_GROUP_MIN_FACE_COUNT
        and max_detector_face_ratio >= YUNET_GROUP_MIN_FACE_RATIO
    ):
        return True

    if (
        len(detector_faces) >= YUNET_PAIR_MIN_FACE_COUNT
        and max_detector_face_ratio >= YUNET_PAIR_MIN_FACE_RATIO
    ):
        return True

    if is_detector_face_source(primary_face.source):
        if not is_plausible_yolo_face(primary_face):
            return False
        if primary_face.is_validated:
            return (
                primary_face.area_ratio >= YUNET_VALIDATED_PERSON_MIN_FACE_RATIO
                or min_dimension >= YUNET_VALIDATED_PERSON_MIN_FACE_PIXELS
            )
        return (
            primary_face.area_ratio >= YUNET_UNVALIDATED_PERSON_MIN_FACE_RATIO
            or min_dimension >= YUNET_UNVALIDATED_PERSON_MIN_FACE_PIXELS
        )

    if not primary_face.is_validated:
        return False

    if primary_face.center_distance_ratio > HAAR_VALIDATED_PERSON_MAX_CENTER_DISTANCE:
        return False

    return (
        primary_face.area_ratio >= HAAR_VALIDATED_PERSON_MIN_FACE_RATIO
        or min_dimension >= HAAR_VALIDATED_PERSON_MIN_FACE_PIXELS
    )


def validate_face_candidate(
    rgb_image: np.ndarray,
    candidate: FaceCandidate,
    detector: FaceLandmarkDetector,
    ear_threshold: float,
    landmark_min_crop_size: int,
    crop_scale: float,
) -> FaceResult | None:
    image_height, image_width = rgb_image.shape[:2]
    crop_bbox = expand_bbox(
        bbox=candidate.bbox,
        image_width=image_width,
        image_height=image_height,
        scale=crop_scale,
    )
    crop_image, resize_scale = prepare_landmark_crop(
        rgb_image=rgb_image,
        crop_bbox=crop_bbox,
        landmark_min_crop_size=landmark_min_crop_size,
    )
    if crop_image.size == 0:
        return None

    crop_height, crop_width = crop_image.shape[:2]
    best_landmarks = None
    for variant in generate_landmark_variants(crop_image):
        variant_height, variant_width = variant.shape[:2]
        landmarks_list = detector.detect(variant)
        best_landmarks = select_best_landmarks(
            landmarks_list=landmarks_list,
            image_width=variant_width,
            image_height=variant_height,
        )
        if best_landmarks is not None:
            crop_image = variant
            crop_height, crop_width = variant_height, variant_width
            break

    if best_landmarks is None and should_try_full_image_landmarks(candidate):
        full_image_height, full_image_width = rgb_image.shape[:2]
        full_image_landmarks = detector.detect(rgb_image)
        best_landmarks = select_best_landmarks_for_candidate(
            landmarks_list=full_image_landmarks,
            candidate_bbox=candidate.bbox,
            image_width=full_image_width,
            image_height=full_image_height,
        )
        if best_landmarks is not None:
            return create_face_result_from_landmarks(
                landmarks=best_landmarks,
                face_id=0,
                image_width=full_image_width,
                image_height=full_image_height,
                ear_threshold=ear_threshold,
                source=f"{candidate.source}_full",
                score=candidate.score,
            )
    if best_landmarks is None:
        return None

    landmark_bbox_in_crop = compute_bbox(best_landmarks, crop_width, crop_height)
    face_bbox = map_bbox_to_original_image(
        bbox=landmark_bbox_in_crop,
        crop_bbox=crop_bbox,
        resize_scale=resize_scale,
        image_width=image_width,
        image_height=image_height,
    )
    left_ear = compute_ear(best_landmarks, LEFT_EYE_INDEXES, crop_width, crop_height)
    right_ear = compute_ear(best_landmarks, RIGHT_EYE_INDEXES, crop_width, crop_height)
    avg_ear = (left_ear + right_ear) / 2.0
    face_area_ratio = compute_area_ratio(face_bbox, image_width, image_height)

    return FaceResult(
        face_id=0,
        ear=avg_ear,
        left_ear=left_ear,
        right_ear=right_ear,
        is_closed=avg_ear is not None and avg_ear < ear_threshold,
        bbox=face_bbox,
        area_ratio=face_area_ratio,
        center_distance_ratio=compute_center_distance_ratio(
            face_bbox, image_width, image_height
        ),
        source=candidate.source,
        score=candidate.score,
        is_validated=True,
    )


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def extract_quality_region(
    rgb_image: np.ndarray,
    primary_face_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if primary_face_bbox is None:
        return rgb_image

    image_height, image_width = rgb_image.shape[:2]
    x1, y1, x2, y2 = primary_face_bbox
    face_width = x2 - x1
    face_height = y2 - y1
    if face_width <= 0 or face_height <= 0:
        return rgb_image

    pad_x = int(round(face_width * 0.15))
    pad_y = int(round(face_height * 0.15))
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(image_width, x2 + pad_x)
    crop_y2 = min(image_height, y2 + pad_y)
    crop = rgb_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return rgb_image
    return crop


def compute_blur_score(
    rgb_image: np.ndarray,
    primary_face_bbox: tuple[int, int, int, int] | None,
) -> float:
    region = extract_quality_region(rgb_image, primary_face_bbox)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    min_var = 15.0
    max_var = 350.0
    normalized = (
        math.log1p(laplacian_variance) - math.log1p(min_var)
    ) / (math.log1p(max_var) - math.log1p(min_var))
    return round(clamp_score(normalized), 3)


def compute_exposure_score(
    rgb_image: np.ndarray,
    primary_face_bbox: tuple[int, int, int, int] | None,
) -> float:
    region = extract_quality_region(rgb_image, primary_face_bbox)
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    normalized_luma = float(np.mean(gray) / 255.0)
    luma_score = 1.0 - min(1.0, abs(normalized_luma - 0.5) / 0.5)
    clip_low_ratio = float(np.mean(gray <= 5))
    clip_high_ratio = float(np.mean(gray >= 250))
    clipping_penalty = min(1.0, (clip_low_ratio + clip_high_ratio) / 0.2)
    score = (0.75 * luma_score) + (0.25 * (1.0 - clipping_penalty))
    return round(clamp_score(score), 3)


def estimate_eyes_open_prob(
    faces: list[FaceResult],
    ear_threshold: float,
    min_face_size: float,
) -> float:
    probabilities: list[float] = []
    for face in faces:
        if not has_reliable_eye_measurement(face, min_face_size):
            continue
        assert face.ear is not None
        # EAR is converted to [0, 1] open-eye probability with a linear ramp.
        probability = clamp_score((face.ear - (ear_threshold - 0.06)) / 0.24)
        probabilities.append(probability)
    if not probabilities:
        return 0.0
    return round(max(probabilities), 3)


def is_primary_face_closed(
    primary_face: FaceResult | None,
    min_face_size: float,
) -> bool:
    if primary_face is None:
        return False
    if not has_reliable_eye_measurement(primary_face, min_face_size):
        return False
    return bool(primary_face.is_closed)


def count_secondary_closed_faces(
    faces: list[FaceResult],
    primary_face_id: int | None,
    min_face_size: float,
) -> int:
    return sum(
        1
        for face in faces
        if face.face_id != primary_face_id
        and has_reliable_eye_measurement(face, min_face_size)
        and face.is_closed
    )


def estimate_usable_score(
    is_person: bool,
    primary_face: FaceResult | None,
    min_face_size: float,
    blur_score: float,
    exposure_score: float,
    eyes_open_prob: float,
    primary_face_closed: bool,
    secondary_closed_face_count: int,
    face_count: int,
) -> float:
    if primary_face is None:
        face_size_score = 0.0
    else:
        min_ref = max(min_face_size, 1e-6)
        face_size_score = clamp_score((primary_face.area_ratio / min_ref) / 4.0)
    person_score = 1.0 if is_person else 0.0
    # Weighted quality fusion (all in [0, 1]); blur matters more than exposure,
    # and face size is intentionally a weak prior.
    score = (
        (EYES_OPEN_WEIGHT * eyes_open_prob)
        + (BLUR_WEIGHT * blur_score)
        + (EXPOSURE_WEIGHT * exposure_score)
        + (FACE_SIZE_WEIGHT * face_size_score)
        + (PERSON_WEIGHT * person_score)
    )
    if primary_face_closed:
        score -= PRIMARY_FACE_CLOSED_PENALTY
    if secondary_closed_face_count > 0:
        score -= min(
            SECONDARY_CLOSED_FACE_PENALTY_CAP,
            SECONDARY_CLOSED_FACE_PENALTY * secondary_closed_face_count,
        )
    if not is_person:
        score -= NON_PERSON_PENALTY

    if face_count >= 4:
        score -= GROUP_FACE_PENALTY

    return round(clamp_score(score), 3)


def decide_photo(
    usable_score: float,
    eyes_open_prob: float,
    blur_score: float,
    exposure_score: float,
    is_person: bool,
    primary_face_closed: bool,
    secondary_closed_face_count: int,
) -> str:
    # Decision thresholds:
    # 1) best: excellent global quality + strong open-eye confidence + no closed-eye penalty.
    # 2) keep: good quality and likely usable.
    # 3) review: borderline quality or uncertainty.
    # 4) reject: quality too low or primary-face eye closure with low score.
    if primary_face_closed:
        return "review" if usable_score >= PRIMARY_FACE_CLOSED_REVIEW_THRESHOLD else "reject"

    if (
        is_person
        and
        usable_score >= 0.88
        and eyes_open_prob >= 0.78
        and blur_score >= 0.68
        and exposure_score >= 0.65
        and secondary_closed_face_count == 0
    ):
        return "best"

    if is_person and usable_score >= 0.64 and eyes_open_prob >= 0.45:
        return "keep"

    if usable_score >= 0.35:
        return "review"

    return "reject"


def build_assessment_reasons(
    decision: str,
    legacy_status: str,
    is_person: bool,
    closed_from_center_faces: bool,
    closed_from_primary_face: bool,
    blur_score: float,
    exposure_score: float,
    eyes_open_prob: float,
    has_open_eye: bool,
    primary_face_closed: bool,
    secondary_closed_face_count: int,
    usable_score: float,
) -> list[str]:
    reasons: list[str] = []
    if not is_person:
        reasons.append("person_confidence_low")

    if primary_face_closed:
        reasons.append("primary_face_closed")
    if secondary_closed_face_count > 0:
        reasons.append(f"secondary_closed_faces={secondary_closed_face_count}")
    if closed_from_center_faces:
        reasons.append("closed_eye_center_priority")
    if closed_from_primary_face:
        reasons.append("closed_eye_primary_face")

    if legacy_status != "normal_face":
        reasons.append(f"legacy_status={legacy_status}")

    if has_open_eye:
        reasons.append("open_eye_detected")
    if blur_score < 0.4:
        reasons.append("blurry_image")
    if exposure_score < 0.35:
        reasons.append("poor_exposure")
    if eyes_open_prob < 0.35:
        reasons.append("eyes_open_probability_low")
    if usable_score < 0.35:
        reasons.append("usable_score_low")

    if decision == "best":
        reasons.append("high_overall_quality")
    elif decision == "keep":
        reasons.append("meets_keep_threshold")
    elif decision == "review":
        reasons.append("manual_review_recommended")
    elif decision == "reject":
        reasons.append("fails_quality_threshold")
    return reasons


def build_photo_assessment(
    image_path: Path,
    legacy_status: str,
    faces: list[FaceResult],
    primary_face: FaceResult | None,
    closed_face_count: int,
    min_ear: float | None,
    blur_score: float,
    exposure_score: float,
    eyes_open_prob: float,
    usable_score: float,
    decision: str,
    is_person: bool,
    closed_from_center_faces: bool,
    closed_from_primary_face: bool,
    has_open_eye: bool,
    primary_face_closed: bool,
    secondary_closed_face_count: int,
) -> PhotoAssessment:
    primary_face_id = primary_face.face_id if primary_face is not None else None
    primary_face_ear = primary_face.ear if primary_face is not None else None
    primary_face_bbox = primary_face.bbox if primary_face is not None else None
    scene_category = "person" if is_person else "non_person"
    reasons = build_assessment_reasons(
        decision=decision,
        legacy_status=legacy_status,
        is_person=is_person,
        closed_from_center_faces=closed_from_center_faces,
        closed_from_primary_face=closed_from_primary_face,
        blur_score=blur_score,
        exposure_score=exposure_score,
        eyes_open_prob=eyes_open_prob,
        has_open_eye=has_open_eye,
        primary_face_closed=primary_face_closed,
        secondary_closed_face_count=secondary_closed_face_count,
        usable_score=usable_score,
    )
    return PhotoAssessment(
        image_path=image_path,
        decision=decision,
        scene_category=scene_category,
        is_person=is_person,
        usable_score=usable_score,
        reasons=reasons,
        legacy_status=legacy_status,
        primary_face_bbox=primary_face_bbox,
        face_count=len(faces),
        closed_face_count=closed_face_count,
        min_ear=min_ear,
        blur_score=blur_score,
        exposure_score=exposure_score,
        eyes_open_prob=eyes_open_prob,
        faces=faces,
        primary_face_id=primary_face_id,
        primary_face_ear=primary_face_ear,
    )


def analyze_image(
    image_path: Path,
    face_candidate_detector: FaceCandidateDetector,
    detector: FaceLandmarkDetector,
    max_faces: int,
    ear_threshold: float,
    min_face_size: float,
    crop_scale: float,
    landmark_min_crop_size: int,
) -> PhotoAssessment:
    rgb_image = load_rgb_image(image_path)
    candidates = face_candidate_detector.detect(rgb_image)
    if not candidates:
        blur_score = compute_blur_score(rgb_image, primary_face_bbox=None)
        exposure_score = compute_exposure_score(rgb_image, primary_face_bbox=None)
        eyes_open_prob = 0.0
        primary_face_closed = False
        secondary_closed_face_count = 0
        usable_score = estimate_usable_score(
            is_person=False,
            primary_face=None,
            min_face_size=min_face_size,
            blur_score=blur_score,
            exposure_score=exposure_score,
            eyes_open_prob=eyes_open_prob,
            primary_face_closed=primary_face_closed,
            secondary_closed_face_count=secondary_closed_face_count,
            face_count=0,
        )
        decision = decide_photo(
            usable_score=usable_score,
            eyes_open_prob=eyes_open_prob,
            blur_score=blur_score,
            exposure_score=exposure_score,
            is_person=False,
            primary_face_closed=primary_face_closed,
            secondary_closed_face_count=secondary_closed_face_count,
        )
        return build_photo_assessment(
            image_path=image_path,
            legacy_status="no_face",
            faces=[],
            primary_face=None,
            closed_face_count=0,
            min_ear=None,
            blur_score=blur_score,
            exposure_score=exposure_score,
            eyes_open_prob=eyes_open_prob,
            usable_score=usable_score,
            decision=decision,
            is_person=False,
            closed_from_center_faces=False,
            closed_from_primary_face=False,
            has_open_eye=False,
            primary_face_closed=primary_face_closed,
            secondary_closed_face_count=secondary_closed_face_count,
        )

    limited_candidates = candidates[:max_faces]
    faces = [
        create_face_result_from_candidate(candidate, face_id=index + 1)
        for index, candidate in enumerate(limited_candidates)
    ]
    primary_face = select_primary_face(faces)
    center_priority_faces = select_center_priority_faces(
        faces,
        limit=CENTER_PRIORITY_FACE_COUNT,
    )
    yolo_priority_faces = select_yolo_priority_faces(
        faces,
        limit=YUNET_PRIORITY_FACE_COUNT,
    )
    center_priority_face_ids = {face.face_id for face in center_priority_faces}
    yolo_priority_face_ids = {face.face_id for face in yolo_priority_faces}
    face_ids_to_validate = sorted(
        center_priority_face_ids
        | yolo_priority_face_ids
        | {primary_face.face_id}
    )

    for face_id in face_ids_to_validate:
        candidate = limited_candidates[face_id - 1]
        validated_face = validate_face_candidate(
            rgb_image=rgb_image,
            candidate=candidate,
            detector=detector,
            ear_threshold=ear_threshold,
            landmark_min_crop_size=landmark_min_crop_size,
            crop_scale=crop_scale,
        )
        if validated_face is None:
            continue
        validated_face.face_id = face_id
        faces[face_id - 1] = validated_face

    validated_faces = [face for face in faces if face.is_validated]
    validated_yolo_faces = [
        face for face in validated_faces if is_detector_face_source(face.source)
    ]
    if validated_yolo_faces:
        validated_faces = validated_yolo_faces
    if validated_faces:
        faces = reindex_faces(validated_faces)
    else:
        # Keep one fallback face so completely failed landmark images still return context.
        # Prefer a YOLO face when available.
        yolo_fallback_faces = select_yolo_priority_faces(faces, limit=1)
        if yolo_fallback_faces:
            fallback_primary_face = yolo_fallback_faces[0]
        else:
            fallback_primary_face = faces[primary_face.face_id - 1]
        faces = reindex_faces([fallback_primary_face])

    primary_face = select_primary_face(faces)
    final_center_priority_faces = select_center_priority_faces(
        faces,
        limit=CENTER_PRIORITY_FACE_COUNT,
    )
    final_center_priority_face_ids = {face.face_id for face in final_center_priority_faces}
    for face in faces:
        face.is_center_priority = face.face_id in final_center_priority_face_ids

    closed_face_count = sum(face.is_closed for face in faces if face.is_validated)
    validated_ears = [face.ear for face in faces if face.ear is not None]
    min_ear = min(validated_ears) if validated_ears else None
    is_person = should_classify_as_person(primary_face, faces)
    has_open_eye = has_reliable_open_eye_face(
        faces=faces,
        ear_threshold=ear_threshold,
        min_face_size=min_face_size,
    )
    closed_from_center_faces = should_classify_closed_eye_from_center_faces(
        faces,
        ear_threshold,
        min_face_size,
    )
    closed_from_primary_face = should_classify_closed_eye(
        primary_face,
        faces,
        ear_threshold,
        min_face_size,
    )
    if not is_person:
        legacy_status = "no_face"
    elif closed_from_center_faces:
        legacy_status = "closed_eye"
    elif closed_from_primary_face:
        legacy_status = "closed_eye"
    elif has_open_eye:
        legacy_status = "normal_face"
    else:
        legacy_status = "face_no_eye"

    primary_face_bbox = primary_face.bbox if primary_face is not None else None
    blur_score = compute_blur_score(rgb_image, primary_face_bbox=primary_face_bbox)
    exposure_score = compute_exposure_score(rgb_image, primary_face_bbox=primary_face_bbox)
    eyes_open_prob = estimate_eyes_open_prob(
        faces=faces,
        ear_threshold=ear_threshold,
        min_face_size=min_face_size,
    )
    primary_face_closed = is_primary_face_closed(primary_face, min_face_size=min_face_size)
    secondary_closed_face_count = count_secondary_closed_faces(
        faces=faces,
        primary_face_id=primary_face.face_id if primary_face is not None else None,
        min_face_size=min_face_size,
    )
    usable_score = estimate_usable_score(
        is_person=is_person,
        primary_face=primary_face,
        min_face_size=min_face_size,
        blur_score=blur_score,
        exposure_score=exposure_score,
        eyes_open_prob=eyes_open_prob,
        primary_face_closed=primary_face_closed,
        secondary_closed_face_count=secondary_closed_face_count,
        face_count=len(faces),
    )
    decision = decide_photo(
        usable_score=usable_score,
        eyes_open_prob=eyes_open_prob,
        blur_score=blur_score,
        exposure_score=exposure_score,
        is_person=is_person,
        primary_face_closed=primary_face_closed,
        secondary_closed_face_count=secondary_closed_face_count,
    )
    return build_photo_assessment(
        image_path=image_path,
        legacy_status=legacy_status,
        faces=faces,
        primary_face=primary_face,
        closed_face_count=closed_face_count,
        min_ear=min_ear,
        blur_score=blur_score,
        exposure_score=exposure_score,
        eyes_open_prob=eyes_open_prob,
        usable_score=usable_score,
        decision=decision,
        is_person=is_person,
        closed_from_center_faces=closed_from_center_faces,
        closed_from_primary_face=closed_from_primary_face,
        has_open_eye=has_open_eye,
        primary_face_closed=primary_face_closed,
        secondary_closed_face_count=secondary_closed_face_count,
    )


def analyze_image_with_timing(
    image_path: Path,
    face_candidate_detector: FaceCandidateDetector,
    detector: FaceLandmarkDetector,
    max_faces: int,
    ear_threshold: float,
    min_face_size: float,
    crop_scale: float,
    landmark_min_crop_size: int,
) -> PhotoAssessment:
    started_at = time.perf_counter()
    result = analyze_image(
        image_path=image_path,
        face_candidate_detector=face_candidate_detector,
        detector=detector,
        max_faces=max_faces,
        ear_threshold=ear_threshold,
        min_face_size=min_face_size,
        crop_scale=crop_scale,
        landmark_min_crop_size=landmark_min_crop_size,
    )
    result.processing_seconds = round(time.perf_counter() - started_at, 3)
    return result


def format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def wrap_annotation_text(text: str, width: int = 72) -> list[str]:
    wrapped = textwrap.wrap(text, width=width)
    return wrapped or [text]


def get_recognition_effects_root(output_dir: Path) -> Path:
    return output_dir / RECOGNITION_EFFECTS_DIRNAME


def should_export_classified(output_mode: str) -> bool:
    return output_mode in {OUTPUT_MODE_CLASSIFIED, OUTPUT_MODE_BOTH}


def should_export_effects(output_mode: str) -> bool:
    return output_mode in {OUTPUT_MODE_EFFECTS, OUTPUT_MODE_BOTH}


def build_recognition_effect_output_path(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    category: str,
) -> Path:
    relative_path = get_relative_image_path(image_path, input_root).with_suffix(".jpg")
    return get_recognition_effects_root(output_dir) / category / relative_path


def measure_text_block(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    font: ImageFont.ImageFont,
    spacing: int = 2,
) -> tuple[int, int]:
    text = "\n".join(lines)
    left, top, right, bottom = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        spacing=spacing,
    )
    return right - left, bottom - top


def draw_text_block(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    lines: list[str],
    *,
    font: ImageFont.ImageFont,
    text_fill: tuple[int, int, int],
    border_fill: tuple[int, int, int],
    spacing: int = 2,
) -> None:
    padding_x = 6
    padding_y = 4
    text_width, text_height = measure_text_block(draw, lines, font=font, spacing=spacing)
    left, top = position
    right = left + text_width + (padding_x * 2)
    bottom = top + text_height + (padding_y * 2)
    draw.rectangle(
        (left, top, right, bottom),
        fill=(0, 0, 0),
        outline=border_fill,
        width=2,
    )
    draw.multiline_text(
        (left + padding_x, top + padding_y),
        "\n".join(lines),
        fill=text_fill,
        font=font,
        spacing=spacing,
    )


def annotate_image(image_path: Path, image_result: PhotoAssessment, output_path: Path) -> None:
    annotated = load_review_image(image_path)
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    image_width, image_height = annotated.size
    summary_color = {
        "reject": (220, 40, 40),
        "review": (240, 170, 40),
        "keep": (40, 180, 90),
        "best": (20, 140, 220),
    }.get(image_result.decision, (255, 255, 255))
    summary_lines = [
        (
            f"decision={image_result.decision} scene={image_result.scene_category} "
            f"person={int(image_result.is_person)}"
        ),
        (
            f"score={image_result.usable_score:.3f} "
            f"eyes_open={image_result.eyes_open_prob:.3f} "
            f"blur={image_result.blur_score:.3f} exp={image_result.exposure_score:.3f}"
        ),
        (
            f"legacy={image_result.legacy_status} faces={image_result.face_count} "
            f"closed={image_result.closed_face_count} "
            f"primary={image_result.primary_face_id or 'n/a'} "
            f"min_ear={format_metric(image_result.min_ear)}"
        ),
    ]
    reasons_text = ", ".join(image_result.reasons) if image_result.reasons else "none"
    summary_lines.extend(wrap_annotation_text(f"reasons={reasons_text}", width=80))
    draw_text_block(
        draw,
        (12, 12),
        summary_lines,
        font=font,
        text_fill=summary_color,
        border_fill=summary_color,
    )

    for face in image_result.faces:
        if face.is_validated and face.is_closed:
            color = (220, 40, 40)
            eye_state = "closed_eye"
        elif face.is_validated and face.ear is not None:
            color = (40, 180, 90)
            eye_state = "normal_eye"
        elif face.is_validated:
            color = (40, 120, 220)
            eye_state = "eye_unknown"
        else:
            color = (40, 120, 220)
            eye_state = "detected_only"

        if face.is_primary:
            width = 5
        elif face.is_center_priority:
            width = 4
        else:
            width = 2

        role_tags: list[str] = []
        if face.is_primary:
            role_tags.append("primary")
        if face.is_center_priority:
            role_tags.append("center_top3")
        if not role_tags:
            role_tags.append("secondary")
        role = "+".join(role_tags)

        draw.rectangle(face.bbox, outline=color, width=width)
        label_lines = [
            f"{role} face={face.face_id} {eye_state}",
            (
                f"src={face.source} det={format_metric(face.score)} "
                f"ear={format_metric(face.ear)}"
            ),
            (
                f"area={face.area_ratio:.4f} "
                f"center={face.center_distance_ratio:.3f}"
            ),
        ]
        if face.left_ear is not None or face.right_ear is not None:
            label_lines.append(
                f"left={format_metric(face.left_ear)} right={format_metric(face.right_ear)}"
            )

        text_width, text_height = measure_text_block(draw, label_lines, font=font)
        label_left = min(max(0, face.bbox[0] + 4), max(0, image_width - text_width - 16))
        preferred_top = face.bbox[1] - text_height - 16
        if preferred_top >= 0:
            label_top = preferred_top
        else:
            label_top = min(face.bbox[3] + 6, max(0, image_height - text_height - 12))
        draw_text_block(
            draw,
            (label_left, label_top),
            label_lines,
            font=font,
            text_fill=color,
            border_fill=color,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, quality=92)


def write_csv(results: list[PhotoAssessment], output_dir: Path) -> Path:
    csv_path = output_dir / "eye_detection_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_path",
                "decision",
                "scene_category",
                "is_person",
                "usable_score",
                "processing_seconds",
                "reasons",
                "legacy_status",
                "face_count",
                "closed_face_count",
                "primary_face_id",
                "primary_face_bbox",
                "primary_face_ear",
                "min_ear",
                "eyes_open_prob",
                "blur_score",
                "exposure_score",
            ]
        )
        for result in results:
            min_ear = "" if result.min_ear is None else f"{result.min_ear:.4f}"
            primary_face_ear = (
                "" if result.primary_face_ear is None else f"{result.primary_face_ear:.4f}"
            )
            primary_face_bbox = (
                "" if result.primary_face_bbox is None else str(result.primary_face_bbox)
            )
            writer.writerow(
                [
                    str(result.image_path),
                    result.decision,
                    result.scene_category,
                    int(result.is_person),
                    f"{result.usable_score:.3f}",
                    f"{result.processing_seconds:.3f}",
                    "|".join(result.reasons),
                    result.legacy_status,
                    result.face_count,
                    result.closed_face_count,
                    result.primary_face_id or "",
                    primary_face_bbox,
                    primary_face_ear,
                    min_ear,
                    f"{result.eyes_open_prob:.3f}",
                    f"{result.blur_score:.3f}",
                    f"{result.exposure_score:.3f}",
                ]
            )
    return csv_path


def get_relative_image_path(image_path: Path, input_root: Path) -> Path:
    if input_root.is_file():
        return Path(image_path.name)
    return image_path.resolve().relative_to(input_root.resolve())


def copy_to_category(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    category: str,
) -> Path:
    relative_path = get_relative_image_path(image_path, input_root)
    destination = output_dir / category / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination)
    return destination


def copy_result_outputs(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    result: PhotoAssessment,
) -> list[Path]:
    return [
        copy_to_category(
            image_path=image_path,
            input_root=input_root,
            output_dir=output_dir,
            category=result.decision,
        )
    ]


def export_result_outputs(
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    result: PhotoAssessment,
    output_mode: str,
) -> dict[str, Path]:
    destinations: dict[str, Path] = {}
    if should_export_classified(output_mode):
        destinations["classified"] = copy_to_category(
            image_path=image_path,
            input_root=input_root,
            output_dir=output_dir,
            category=result.decision,
        )
    if should_export_effects(output_mode):
        effect_output_path = build_recognition_effect_output_path(
            image_path=image_path,
            input_root=input_root,
            output_dir=output_dir,
            category=result.decision,
        )
        annotate_image(
            image_path=image_path,
            image_result=result,
            output_path=effect_output_path,
        )
        destinations["effects"] = effect_output_path
    return destinations


def format_output_destinations(destinations: dict[str, Path]) -> str:
    parts: list[str] = []
    if "classified" in destinations:
        parts.append(f"classified={destinations['classified']}")
    if "effects" in destinations:
        parts.append(f"effects={destinations['effects']}")
    return " | ".join(parts) if parts else "none"


def ensure_decision_output_structure(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for category_name in DECISION_NAMES:
        (output_dir / category_name).mkdir(parents=True, exist_ok=True)


def ensure_output_structure(output_dir: Path, output_mode: str = OUTPUT_MODE_CLASSIFIED) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if should_export_classified(output_mode):
        ensure_decision_output_structure(output_dir)
    if should_export_effects(output_mode):
        ensure_decision_output_structure(get_recognition_effects_root(output_dir))


def main() -> int:
    args = parse_args()
    ensure_dependencies()
    run_started_at = time.perf_counter()

    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.expanduser().resolve()
    exclude_dirs = [output_dir] if input_path.is_dir() else []

    image_paths = iter_images(input_path, exclude_dirs=exclude_dirs)
    if not image_paths:
        print("No supported images found.", file=sys.stderr)
        return 1
    ensure_raw_support(image_paths)

    ensure_output_structure(output_dir, args.output_mode)

    face_candidate_detector = create_face_candidate_detector(args)
    detector = create_detector(
        max_faces=max(args.max_faces, CENTER_PRIORITY_FACE_COUNT),
        model_path=args.model_path,
    )

    results: list[PhotoAssessment] = []
    try:
        for image_path in image_paths:
            result = analyze_image_with_timing(
                image_path=image_path,
                face_candidate_detector=face_candidate_detector,
                detector=detector,
                max_faces=args.max_faces,
                ear_threshold=args.ear_threshold,
                min_face_size=args.min_face_size,
                crop_scale=args.crop_scale,
                landmark_min_crop_size=args.landmark_min_crop_size,
            )
            results.append(result)
            destinations = export_result_outputs(
                image_path=image_path,
                input_root=input_path,
                output_dir=output_dir,
                result=result,
                output_mode=args.output_mode,
            )
            primary_ear_text = (
                "n/a" if result.primary_face_ear is None else f"{result.primary_face_ear:.3f}"
            )
            print(
                f"{result.decision:7} score={result.usable_score:.3f} "
                f"scene={result.scene_category:11} legacy={result.legacy_status:12} "
                f"faces={result.face_count:<2} "
                f"closed={result.closed_face_count:<2} primary={result.primary_face_id or 'n/a':<3} "
                f"primary_ear={primary_ear_text} time={result.processing_seconds:.3f}s "
                f"dest={format_output_destinations(destinations)}"
            )
    finally:
        face_candidate_detector.close()
        detector.close()

    csv_path = write_csv(results, output_dir)
    total_processing_seconds = round(time.perf_counter() - run_started_at, 3)
    reject_count = sum(result.decision == "reject" for result in results)
    review_count = sum(result.decision == "review" for result in results)
    keep_count = sum(result.decision == "keep" for result in results)
    best_count = sum(result.decision == "best" for result in results)
    non_person_count = sum(not result.is_person for result in results)
    no_face_count = sum(result.legacy_status == "no_face" for result in results)
    face_no_eye_count = sum(result.legacy_status == "face_no_eye" for result in results)
    closed_eye_count = sum(result.legacy_status == "closed_eye" for result in results)
    normal_face_count = sum(result.legacy_status == "normal_face" for result in results)
    print(f"\nProcessed {len(results)} image(s).")
    print(f"Total processing time: {total_processing_seconds:.3f}s")
    print(f"Average time per image: {total_processing_seconds / max(len(results), 1):.3f}s")
    print(f"Decision reject: {reject_count}")
    print(f"Decision review: {review_count}")
    print(f"Decision keep: {keep_count}")
    print(f"Decision best: {best_count}")
    print(f"Non-person sent to reject: {non_person_count}")
    print(
        "Legacy status (debug) "
        f"no_face={no_face_count} face_no_eye={face_no_eye_count} "
        f"closed_eye={closed_eye_count} normal_face={normal_face_count}"
    )
    print(f"CSV report: {csv_path}")
    print(f"Output mode: {args.output_mode}")
    if should_export_classified(args.output_mode):
        print(f"Classified files: {output_dir}")
    if should_export_effects(args.output_mode):
        print(f"Recognition effects: {get_recognition_effects_root(output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
