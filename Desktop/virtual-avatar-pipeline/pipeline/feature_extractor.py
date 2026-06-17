"""
MediaPipe FaceLandmarker (Tasks API) 기반 얼굴 특징 추출기
mediapipe 0.10+ 전용

모델 파일(face_landmarker.task)은 최초 실행 시 자동 다운로드됩니다.
"""

import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from dataclasses import dataclass
from pathlib import Path
from PIL import Image


_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
_MODEL_PATH = Path(__file__).parent / "face_landmarker.task"

FEATURE_SCHEMA_VERSION = "1.5"

CORE_FEATURE_FIELDS = [
    "eye_aspect_ratio",
    "eye_distance_ratio",
    "face_width_height_ratio",
    "nose_height_ratio",
    "nose_width_ratio",
    "mouth_width_ratio",
    "jaw_width_ratio",
    "forehead_ratio",
    "chin_ratio",
]

EXTENDED_FEATURE_FIELDS = [
    "eye_width_ratio",
    "eye_height_ratio",
    "eye_area_ratio",
    "eye_slant",
    "eye_tail_height_delta",
    "eye_center_y_ratio",
    "eye_front_height_delta",
    "eye_front_flatness",
    "upper_lid_flatness",
    "lower_lid_flatness",
    "top_lid_down_score",
    "lower_lid_up_score",
    "iris_to_eye_width_ratio",
    "iris_to_eye_height_ratio",
    "brow_distance_ratio",
    "brow_height_ratio",
    "brow_angle",
    "brow_width_ratio",
    "brow_eye_gap_ratio",
    "nose_tip_y_ratio",
    "nose_bridge_length_ratio",
    "philtrum_length_ratio",
    "nose_to_mouth_center_distance_ratio",
    "nose_under_curve_proxy",
    "mouth_height_ratio",
    "mouth_aspect_ratio",
    "mouth_corner_slope",
    "smile_score_geometry",
    "mouth_open_score_geometry",
    "mouth_center_y_ratio",
    "mouth_to_nose_distance_ratio",
    "mouth_to_chin_distance_ratio",
    "mouth_corner_y_delta",
    "upper_lip_height_ratio",
    "lower_lip_height_ratio",
    "cheek_width_ratio",
    "lower_face_width_ratio",
    "chin_width_ratio",
    "chin_length_ratio",
    "face_taper_ratio",
    "cheek_jaw_delta_ratio",
    "chin_taper_ratio",
    "chin_pointedness_score",
    "jaw_angle_score",
    "jaw_sharpness_score",
    "lower_face_ratio",
    "landmark_bbox_area_ratio",
    "landmark_center_offset",
    "face_roll_proxy",
    "face_yaw_proxy",
    "face_pitch_proxy",
    "face_side_width_asymmetry",
    "eye_open_left_ratio",
    "eye_open_right_ratio",
    "eye_shape_asymmetry",
    "mouth_corner_asymmetry",
    "mouth_expression_confidence",
    "nose_mouth_alignment_offset",
    "landmark_quality_score",
]

# landmark index map (mediapipe canonical face mesh 기준)
_LM = {
    "left_eye_outer":  33,
    "left_eye_inner":  133,
    "left_eye_top":    159,
    "left_eye_bottom": 145,
    "right_eye_outer":  263,
    "right_eye_inner":  362,
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    "nose_tip":    1,
    "nose_bridge": 168,
    "nose_left":   129,
    "nose_right":  358,
    "mouth_left":   61,
    "mouth_right":  291,
    # Inner lip landmarks used for geometry-only mouth height/open metrics.
    "mouth_upper":  13,
    "mouth_lower":  14,
    # Brow landmarks are diagnostic only; hair/occlusion can make them noisy.
    "left_brow_inner":  55,
    "left_brow_outer":  70,
    "left_brow_top":    65,
    "right_brow_inner": 285,
    "right_brow_outer": 300,
    "right_brow_top":   295,
    "forehead":    10,
    "chin":       152,
    "left_cheek":  234,
    "right_cheek": 454,
    "left_jaw":    172,
    "right_jaw":   397,
    # Chin-adjacent face oval landmarks used only for diagnostic chin width.
    "left_chin_side":  148,
    "right_chin_side": 377,
}


@dataclass
class FaceFeatureVector:
    eye_aspect_ratio: float
    eye_distance_ratio: float
    face_width_height_ratio: float
    nose_height_ratio: float
    nose_width_ratio: float
    mouth_width_ratio: float
    jaw_width_ratio: float
    forehead_ratio: float
    chin_ratio: float
    eye_width_ratio: float | None = None
    eye_height_ratio: float | None = None
    eye_area_ratio: float | None = None
    eye_slant: float | None = None
    eye_tail_height_delta: float | None = None
    eye_center_y_ratio: float | None = None
    eye_front_height_delta: float | None = None
    eye_front_flatness: float | None = None
    upper_lid_flatness: float | None = None
    lower_lid_flatness: float | None = None
    top_lid_down_score: float | None = None
    lower_lid_up_score: float | None = None
    iris_to_eye_width_ratio: float | None = None
    iris_to_eye_height_ratio: float | None = None
    brow_distance_ratio: float | None = None
    brow_height_ratio: float | None = None
    brow_angle: float | None = None
    brow_width_ratio: float | None = None
    brow_eye_gap_ratio: float | None = None
    nose_tip_y_ratio: float | None = None
    nose_bridge_length_ratio: float | None = None
    philtrum_length_ratio: float | None = None
    nose_to_mouth_center_distance_ratio: float | None = None
    nose_under_curve_proxy: float | None = None
    mouth_height_ratio: float | None = None
    mouth_aspect_ratio: float | None = None
    mouth_corner_slope: float | None = None
    smile_score_geometry: float | None = None
    mouth_open_score_geometry: float | None = None
    mouth_center_y_ratio: float | None = None
    mouth_to_nose_distance_ratio: float | None = None
    mouth_to_chin_distance_ratio: float | None = None
    mouth_corner_y_delta: float | None = None
    upper_lip_height_ratio: float | None = None
    lower_lip_height_ratio: float | None = None
    cheek_width_ratio: float | None = None
    lower_face_width_ratio: float | None = None
    chin_width_ratio: float | None = None
    chin_length_ratio: float | None = None
    face_taper_ratio: float | None = None
    cheek_jaw_delta_ratio: float | None = None
    chin_taper_ratio: float | None = None
    chin_pointedness_score: float | None = None
    jaw_angle_score: float | None = None
    jaw_sharpness_score: float | None = None
    lower_face_ratio: float | None = None
    landmark_bbox_area_ratio: float | None = None
    landmark_center_offset: float | None = None
    face_roll_proxy: float | None = None
    face_yaw_proxy: float | None = None
    face_pitch_proxy: float | None = None
    face_side_width_asymmetry: float | None = None
    eye_open_left_ratio: float | None = None
    eye_open_right_ratio: float | None = None
    eye_shape_asymmetry: float | None = None
    mouth_corner_asymmetry: float | None = None
    mouth_expression_confidence: float | None = None
    nose_mouth_alignment_offset: float | None = None
    landmark_quality_score: float | None = None

    def to_dict(
        self,
        include_extended: bool = True,
        drop_none: bool = False,
    ) -> dict:
        names = self.field_names(include_extended=include_extended)
        result = {name: getattr(self, name) for name in names}
        if drop_none:
            result = {name: value for name, value in result.items() if value is not None}
        return result

    def to_array(self, include_extended: bool = False) -> np.ndarray:
        values = []
        for name in self.field_names(include_extended=include_extended):
            value = getattr(self, name)
            values.append(np.nan if value is None else value)
        return np.array(values, dtype=np.float32)

    @classmethod
    def field_names(cls, include_extended: bool = False) -> list[str]:
        names = list(CORE_FEATURE_FIELDS)
        if include_extended:
            names.extend(EXTENDED_FEATURE_FIELDS)
        return names


def extract_features(
    image: "Image.Image | str | np.ndarray",
    min_confidence: float = 0.4,
) -> "FaceFeatureVector | None":
    img_rgb = _to_rgb(image)
    lm = _detect_landmarks(img_rgb, min_confidence)
    if lm is None:
        return None
    return _compute_features(lm, img_rgb.shape)


def visualize_landmarks(
    image: "Image.Image | str | np.ndarray",
    save_path: "str | None" = None,
) -> Image.Image:
    img_rgb = _to_rgb(image)
    lm = _detect_landmarks(img_rgb, 0.3)
    if lm is None:
        return Image.fromarray(img_rgb)

    h, w = img_rgb.shape[:2]
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for key, idx in _LM.items():
        pt = lm[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        cv2.circle(vis, (x, y), 3, (0, 255, 0), -1)
        cv2.putText(vis, key[:3], (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)

    img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    if save_path:
        img.save(save_path)
    return img


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_model():
    if not _MODEL_PATH.exists():
        print(f"[MediaPipe] 모델 다운로드 중... ({_MODEL_URL})")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print(f"[MediaPipe] 모델 저장됨: {_MODEL_PATH}")


def _to_rgb(image) -> np.ndarray:
    if isinstance(image, str):
        try:
            return np.array(Image.open(image).convert("RGB"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Image path not found: {image}") from exc
        except OSError as exc:
            raise OSError(f"Unable to read image: {image}") from exc
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    # BGR ndarray
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detect_landmarks(img_rgb: np.ndarray, min_confidence: float):
    _ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=min_confidence,
        min_face_presence_confidence=min_confidence,
    )
    detector = mp_vision.FaceLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]  # list of NormalizedLandmark


def _pt(lm, key: str, h: int, w: int) -> np.ndarray:
    l = lm[_LM[key]]
    return np.array([l.x * w, l.y * h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _slope(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float((b[1] - a[1]) / ((b[0] - a[0]) + eps))


def _angle_score(a: np.ndarray, center: np.ndarray, b: np.ndarray, eps: float) -> float:
    va = a - center
    vb = b - center
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + eps
    cos_theta = float(np.clip(np.dot(va, vb) / denom, -1.0, 1.0))
    return float(np.arccos(cos_theta) / np.pi)


def _landmark_bbox_metrics(points: list[np.ndarray], shape: tuple, eps: float) -> tuple[float, float]:
    h, w = shape[:2]
    xy = np.array(points, dtype=np.float32)
    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    bbox_area_ratio = float(((x_max - x_min) * (y_max - y_min)) / ((w * h) + eps))

    bbox_center = np.array([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])
    image_center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    half_diag = np.linalg.norm(np.array([w, h], dtype=np.float32)) / 2.0
    center_offset = float(_dist(bbox_center, image_center) / (half_diag + eps))
    return bbox_area_ratio, center_offset


def _compute_features(lm, shape: tuple) -> FaceFeatureVector:
    h, w = shape[:2]
    g = lambda k: _pt(lm, k, h, w)

    face_w = _dist(g("left_cheek"), g("right_cheek"))
    face_h = _dist(g("forehead"), g("chin"))
    eps = 1e-6

    left_eye_width = _dist(g("left_eye_outer"), g("left_eye_inner"))
    right_eye_width = _dist(g("right_eye_outer"), g("right_eye_inner"))
    left_eye_height = _dist(g("left_eye_top"), g("left_eye_bottom"))
    right_eye_height = _dist(g("right_eye_top"), g("right_eye_bottom"))

    left_ear = left_eye_height / (left_eye_width + eps)
    right_ear = right_eye_height / (right_eye_width + eps)
    eye_ar = (left_ear + right_ear) / 2.0

    left_center = (g("left_eye_outer") + g("left_eye_inner")) / 2.0
    right_center = (g("right_eye_outer") + g("right_eye_inner")) / 2.0
    eye_dist_ratio = _dist(left_center, right_center) / (face_w + eps)

    nose_h = _dist(g("nose_bridge"), g("nose_tip")) / (face_h + eps)
    nose_w = _dist(g("nose_left"), g("nose_right")) / (face_w + eps)
    mouth_w_px = _dist(g("mouth_left"), g("mouth_right"))
    mouth_w = mouth_w_px / (face_w + eps)
    jaw_w = _dist(g("left_jaw"), g("right_jaw")) / (face_w + eps)

    nose_tip_y = g("nose_tip")[1]
    forehead_y = g("forehead")[1]
    chin_y = g("chin")[1]
    forehead_ratio = (nose_tip_y - forehead_y) / (face_h + eps)
    chin_ratio = (chin_y - nose_tip_y) / (face_h + eps)

    eye_width = (left_eye_width + right_eye_width) / 2.0
    eye_height = (left_eye_height + right_eye_height) / 2.0
    eye_area = (
        (left_eye_width * left_eye_height)
        + (right_eye_width * right_eye_height)
    ) / 2.0
    left_eye_slant = (g("left_eye_inner")[1] - g("left_eye_outer")[1]) / (
        left_eye_width + eps
    )
    right_eye_slant = (g("right_eye_inner")[1] - g("right_eye_outer")[1]) / (
        right_eye_width + eps
    )
    eye_slant = (left_eye_slant + right_eye_slant) / 2.0
    eye_tail_delta = (
        (g("left_eye_inner")[1] - g("left_eye_outer")[1])
        + (g("right_eye_inner")[1] - g("right_eye_outer")[1])
    ) / 2.0

    mouth_height = _dist(g("mouth_upper"), g("mouth_lower"))
    mouth_aspect_ratio = mouth_height / (mouth_w_px + eps)
    mouth_center = (g("mouth_upper") + g("mouth_lower")) / 2.0
    mouth_mid_y = mouth_center[1]
    avg_corner_y = (g("mouth_left")[1] + g("mouth_right")[1]) / 2.0
    eye_center = (left_center + right_center) / 2.0
    left_brow_center = (g("left_brow_inner") + g("left_brow_outer")) / 2.0
    right_brow_center = (g("right_brow_inner") + g("right_brow_outer")) / 2.0
    brow_center = (left_brow_center + right_brow_center) / 2.0
    left_brow_width = _dist(g("left_brow_inner"), g("left_brow_outer"))
    right_brow_width = _dist(g("right_brow_inner"), g("right_brow_outer"))
    left_brow_angle = _slope(g("left_brow_inner"), g("left_brow_outer"), eps)
    right_brow_angle = _slope(g("right_brow_inner"), g("right_brow_outer"), eps)
    landmark_bbox_area_ratio, landmark_center_offset = _landmark_bbox_metrics(
        [_pt(lm, key, h, w) for key in _LM],
        shape,
        eps,
    )

    # These diagnostic features are intentionally not wired to avatar parameters.
    # eye_area_ratio is approximate: mean eye width * height, not polygon area.
    eye_center_y_ratio = (eye_center[1] - forehead_y) / (face_h + eps)
    eye_front_height_delta = (
        ((g("left_eye_inner")[1] + g("right_eye_inner")[1]) / 2.0) - eye_center[1]
    ) / (face_h + eps)
    brow_distance_ratio = _dist(g("left_brow_inner"), g("right_brow_inner")) / (
        face_w + eps
    )
    brow_height_ratio = (eye_center[1] - brow_center[1]) / (face_h + eps)
    brow_angle = (left_brow_angle + right_brow_angle) / 2.0
    brow_width_ratio = ((left_brow_width + right_brow_width) / 2.0) / (face_w + eps)
    brow_eye_gap_ratio = (
        ((g("left_eye_top")[1] - left_brow_center[1])
        + (g("right_eye_top")[1] - right_brow_center[1])) / 2.0
    ) / (face_h + eps)
    nose_tip_y_ratio = (nose_tip_y - forehead_y) / (face_h + eps)
    nose_bridge_length_ratio = _dist(g("nose_bridge"), g("nose_tip")) / (face_h + eps)
    philtrum_length_ratio = _dist(g("nose_tip"), g("mouth_upper")) / (face_h + eps)
    nose_to_mouth_center_distance_ratio = _dist(g("nose_tip"), mouth_center) / (
        face_h + eps
    )
    # Nose_UnderNose candidate proxy: vertical under-nose gap normalized by nose width.
    nose_width_px = _dist(g("nose_left"), g("nose_right"))
    nose_under_curve_proxy = (g("mouth_upper")[1] - g("nose_tip")[1]) / (
        nose_width_px + eps
    )
    # mouth_height_ratio represents open amount, not the Mouth_Height parameter.
    mouth_center_y_ratio = (mouth_center[1] - forehead_y) / (face_h + eps)
    mouth_to_nose_distance_ratio = _dist(mouth_center, g("nose_tip")) / (face_h + eps)
    mouth_to_chin_distance_ratio = _dist(mouth_center, g("chin")) / (face_h + eps)
    mouth_corner_y_delta = (g("mouth_right")[1] - g("mouth_left")[1]) / (
        face_h + eps
    )
    upper_lip_height_ratio = abs(g("mouth_upper")[1] - mouth_center[1]) / (
        face_h + eps
    )
    lower_lip_height_ratio = abs(g("mouth_lower")[1] - mouth_center[1]) / (
        face_h + eps
    )
    lower_face_width = _dist(g("left_jaw"), g("right_jaw"))
    chin_width = _dist(g("left_chin_side"), g("right_chin_side"))
    cheek_width_ratio = face_w / (face_h + eps)
    lower_face_width_ratio = lower_face_width / (face_h + eps)
    chin_width_ratio = chin_width / (face_h + eps)
    chin_length_ratio = _dist(mouth_center, g("chin")) / (face_h + eps)
    face_taper_ratio = lower_face_width / (face_w + eps)
    cheek_jaw_delta_ratio = (face_w - lower_face_width) / (face_h + eps)
    chin_taper_ratio = chin_width / (lower_face_width + eps)
    chin_pointedness_score = 1.0 - chin_taper_ratio
    jaw_angle_score = _angle_score(g("left_jaw"), g("chin"), g("right_jaw"), eps)
    jaw_sharpness_score = 1.0 - jaw_angle_score
    # Diagnostic alias with the same basis as chin_ratio.
    lower_face_ratio = chin_ratio
    # Roll proxy helps judge reliability of slope-based diagnostic features.
    face_roll_proxy = _slope(left_center, right_center, eps)
    # Reliability diagnostics are broad 2D landmark proxies, not avatar controls.
    face_yaw_proxy = (g("nose_tip")[0] - eye_center[0]) / (face_w + eps)
    face_pitch_proxy = (
        (g("nose_tip")[1] - eye_center[1])
        - (mouth_center[1] - g("nose_tip")[1])
    ) / (face_h + eps)
    face_side_width_asymmetry = abs(
        _dist(g("nose_tip"), g("left_cheek"))
        - _dist(g("nose_tip"), g("right_cheek"))
    ) / (face_w + eps)
    eye_open_left_ratio = left_ear
    eye_open_right_ratio = right_ear
    eye_shape_asymmetry = abs(left_ear - right_ear) / (eye_ar + eps)
    mouth_corner_asymmetry = abs(g("mouth_right")[1] - g("mouth_left")[1]) / (
        mouth_w_px + eps
    )
    mouth_open_score = float(np.clip(mouth_aspect_ratio / 0.5, 0.0, 1.0))
    smile_expression_score = abs((mouth_mid_y - avg_corner_y) / (face_h + eps)) / 0.08
    mouth_expression_confidence = 1.0 - float(np.clip(
        max(
            mouth_open_score,
            smile_expression_score,
            mouth_corner_asymmetry / 0.15,
        ),
        0.0,
        1.0,
    ))
    nose_mouth_alignment_offset = abs(g("nose_tip")[0] - mouth_center[0]) / (
        face_w + eps
    )

    center_penalty = float(np.clip(landmark_center_offset / 0.35, 0.0, 1.0))
    roll_penalty = float(np.clip(abs(face_roll_proxy) / 0.20, 0.0, 1.0))
    yaw_penalty = float(np.clip(abs(face_yaw_proxy) / 0.20, 0.0, 1.0))
    pitch_penalty = float(np.clip(abs(face_pitch_proxy) / 0.25, 0.0, 1.0))
    side_asymmetry_penalty = float(np.clip(
        face_side_width_asymmetry / 0.25,
        0.0,
        1.0,
    ))
    eye_asymmetry_penalty = float(np.clip(eye_shape_asymmetry / 0.50, 0.0, 1.0))
    mouth_expression_penalty = 1.0 - mouth_expression_confidence
    nose_mouth_penalty = float(np.clip(
        nose_mouth_alignment_offset / 0.15,
        0.0,
        1.0,
    ))
    if 0.04 <= landmark_bbox_area_ratio <= 0.35:
        bbox_penalty = 0.0
    elif landmark_bbox_area_ratio < 0.04:
        bbox_penalty = float(np.clip((0.04 - landmark_bbox_area_ratio) / 0.04, 0.0, 1.0))
    else:
        bbox_penalty = float(np.clip((landmark_bbox_area_ratio - 0.35) / 0.35, 0.0, 1.0))
    landmark_quality_score = float(np.clip(
        1.0 - max(
            center_penalty,
            roll_penalty,
            yaw_penalty,
            pitch_penalty,
            side_asymmetry_penalty,
            eye_asymmetry_penalty,
            mouth_expression_penalty,
            nose_mouth_penalty,
            bbox_penalty,
        ),
        0.0,
        1.0,
    ))

    return FaceFeatureVector(
        eye_aspect_ratio=float(eye_ar),
        eye_distance_ratio=float(eye_dist_ratio),
        face_width_height_ratio=float(face_w / (face_h + eps)),
        nose_height_ratio=float(nose_h),
        nose_width_ratio=float(nose_w),
        mouth_width_ratio=float(mouth_w),
        jaw_width_ratio=float(jaw_w),
        forehead_ratio=float(forehead_ratio),
        chin_ratio=float(chin_ratio),
        eye_width_ratio=float(eye_width / (face_w + eps)),
        eye_height_ratio=float(eye_height / (face_h + eps)),
        eye_area_ratio=float(eye_area / ((face_w * face_h) + eps)),
        eye_slant=float(eye_slant),
        eye_tail_height_delta=float(eye_tail_delta / (face_h + eps)),
        eye_center_y_ratio=float(eye_center_y_ratio),
        eye_front_height_delta=float(eye_front_height_delta),
        eye_front_flatness=None,
        upper_lid_flatness=None,
        lower_lid_flatness=None,
        top_lid_down_score=None,
        lower_lid_up_score=None,
        iris_to_eye_width_ratio=None,
        iris_to_eye_height_ratio=None,
        brow_distance_ratio=float(brow_distance_ratio),
        brow_height_ratio=float(brow_height_ratio),
        brow_angle=float(brow_angle),
        brow_width_ratio=float(brow_width_ratio),
        brow_eye_gap_ratio=float(brow_eye_gap_ratio),
        nose_tip_y_ratio=float(nose_tip_y_ratio),
        nose_bridge_length_ratio=float(nose_bridge_length_ratio),
        philtrum_length_ratio=float(philtrum_length_ratio),
        nose_to_mouth_center_distance_ratio=float(nose_to_mouth_center_distance_ratio),
        nose_under_curve_proxy=float(nose_under_curve_proxy),
        mouth_height_ratio=float(mouth_height / (face_h + eps)),
        mouth_aspect_ratio=float(mouth_aspect_ratio),
        mouth_corner_slope=float(
            (g("mouth_right")[1] - g("mouth_left")[1]) / (mouth_w_px + eps)
        ),
        smile_score_geometry=float((mouth_mid_y - avg_corner_y) / (face_h + eps)),
        mouth_open_score_geometry=float(mouth_open_score),
        mouth_center_y_ratio=float(mouth_center_y_ratio),
        mouth_to_nose_distance_ratio=float(mouth_to_nose_distance_ratio),
        mouth_to_chin_distance_ratio=float(mouth_to_chin_distance_ratio),
        mouth_corner_y_delta=float(mouth_corner_y_delta),
        upper_lip_height_ratio=float(upper_lip_height_ratio),
        lower_lip_height_ratio=float(lower_lip_height_ratio),
        cheek_width_ratio=float(cheek_width_ratio),
        lower_face_width_ratio=float(lower_face_width_ratio),
        chin_width_ratio=float(chin_width_ratio),
        chin_length_ratio=float(chin_length_ratio),
        face_taper_ratio=float(face_taper_ratio),
        cheek_jaw_delta_ratio=float(cheek_jaw_delta_ratio),
        chin_taper_ratio=float(chin_taper_ratio),
        chin_pointedness_score=float(chin_pointedness_score),
        jaw_angle_score=float(jaw_angle_score),
        jaw_sharpness_score=float(jaw_sharpness_score),
        lower_face_ratio=float(lower_face_ratio),
        landmark_bbox_area_ratio=float(landmark_bbox_area_ratio),
        landmark_center_offset=float(landmark_center_offset),
        face_roll_proxy=float(face_roll_proxy),
        face_yaw_proxy=float(face_yaw_proxy),
        face_pitch_proxy=float(face_pitch_proxy),
        face_side_width_asymmetry=float(face_side_width_asymmetry),
        eye_open_left_ratio=float(eye_open_left_ratio),
        eye_open_right_ratio=float(eye_open_right_ratio),
        eye_shape_asymmetry=float(eye_shape_asymmetry),
        mouth_corner_asymmetry=float(mouth_corner_asymmetry),
        mouth_expression_confidence=float(mouth_expression_confidence),
        nose_mouth_alignment_offset=float(nose_mouth_alignment_offset),
        landmark_quality_score=float(landmark_quality_score),
    )
