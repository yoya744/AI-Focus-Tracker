"""MediaPipe Face Landmarker による EAR・頭部姿勢の取得。"""

from __future__ import annotations

import math
import time
import urllib.request
import hashlib
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
)
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode
from mediapipe.tasks.python.vision import drawing_utils as mp_drawing_utils
from mediapipe.tasks.python.vision import drawing_styles as mp_drawing_styles

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

FACE_3D_MODEL = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
    ],
    dtype=np.float64,
)

FACE_LANDMARK_IDX = [1, 152, 263, 33, 287, 57]

_NUM_LANDMARKS = 468
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
_CACHE_DIR = Path.home() / ".cache" / "ai_focus_tracker"
_MODEL_PATH = _CACHE_DIR / "face_landmarker.task"


def _has_expected_model_hash(path: Path) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest() == _MODEL_SHA256


def _ensure_model() -> Path:
    if _MODEL_PATH.exists() and _has_expected_model_hash(_MODEL_PATH):
        return _MODEL_PATH
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _MODEL_PATH.with_suffix(_MODEL_PATH.suffix + ".tmp")
    try:
        with urllib.request.urlopen(_MODEL_URL, timeout=30) as resp, open(tmp_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        if not _has_expected_model_hash(tmp_path):
            raise RuntimeError("Downloaded face landmarker model failed SHA-256 verification")
        tmp_path.replace(_MODEL_PATH)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download face landmarker model: {e}") from e
    return _MODEL_PATH


class FaceAnalyzer:
    """カメラ映像から EAR と頭部姿勢 (yaw/pitch/roll) を取得する。"""

    def __init__(self) -> None:
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode.VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._contour_style = mp_drawing_styles.get_default_face_mesh_contours_style()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._landmarker.close()
            raise RuntimeError("Camera not available")

        self._start_time = time.monotonic()

    def get_frame(self) -> tuple[np.ndarray | None, dict | None]:
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        results = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not results.face_landmarks:
            return frame, None

        face_landmarks = results.face_landmarks[0]
        landmarks = np.array(
            [[lm.x * w, lm.y * h] for lm in face_landmarks[:_NUM_LANDMARKS]],
            dtype=np.float64,
        )

        mp_drawing_utils.draw_landmarks(
            frame,
            face_landmarks[:_NUM_LANDMARKS],
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._contour_style,
        )

        ear_left = self._calc_ear(landmarks, LEFT_EYE_IDX)
        ear_right = self._calc_ear(landmarks, RIGHT_EYE_IDX)
        ear = float((ear_left + ear_right) / 2.0)

        yaw, pitch, roll = self._calc_head_pose(landmarks, frame.shape)

        return frame, {
            "ear": ear,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "landmarks": landmarks,
        }

    def _calc_ear(self, landmarks: np.ndarray, idx: list[int]) -> float:
        points = landmarks[idx]
        p1, p2, p3, p4, p5, p6 = points
        v1 = float(np.linalg.norm(p2 - p6))
        v2 = float(np.linalg.norm(p3 - p5))
        v3 = float(np.linalg.norm(p1 - p4))
        if v3 < 1e-6:
            return 0.0
        return (v1 + v2) / (2.0 * v3)

    def _calc_head_pose(
        self,
        landmarks: np.ndarray,
        frame_shape: tuple[int, int, int],
    ) -> tuple[float, float, float]:
        h, w = frame_shape[0], frame_shape[1]
        focal_length = float(w)
        cx = w / 2.0
        cy = h / 2.0

        camera_matrix = np.array(
            [
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = landmarks[FACE_LANDMARK_IDX].astype(np.float64)
        success, rvec, _tvec = cv2.solvePnP(
            FACE_3D_MODEL,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        return self._rmat_to_euler_deg(rmat)

    @staticmethod
    def _rmat_to_euler_deg(rmat: np.ndarray) -> tuple[float, float, float]:
        """回転行列から (yaw, pitch, roll) を度で返す。右向き・上向き・左傾きが正。"""
        sy = math.sqrt(float(rmat[0, 0] ** 2 + rmat[1, 0] ** 2))
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(float(rmat[2, 1]), float(rmat[2, 2]))
            yaw = math.atan2(float(-rmat[2, 0]), sy)
            roll = math.atan2(float(rmat[1, 0]), float(rmat[0, 0]))
        else:
            pitch = math.atan2(float(-rmat[1, 2]), float(rmat[1, 1]))
            yaw = math.atan2(float(-rmat[2, 0]), sy)
            roll = 0.0

        return (
            math.degrees(yaw),
            math.degrees(pitch),
            math.degrees(roll),
        )

    def release(self) -> None:
        self.cap.release()
        self._landmarker.close()


_QUIT_KEYS = (27, ord("q"), ord("Q"))  # Esc, q, Q


def _poll_quit_key() -> bool:
    """OpenCV ウィンドウがフォーカスされているときのキー入力を確認する。"""
    return (cv2.waitKey(1) & 0xFF) in _QUIT_KEYS


if __name__ == "__main__":
    import sys

    analyzer = FaceAnalyzer()
    print(
        "Face analyzer started.\n"
        "  停止: 映像ウィンドウを選択して Esc または q を押す\n"
        "        （Ctrl+C でも終了できます）",
        file=sys.stderr,
    )
    try:
        while True:
            frame, result = analyzer.get_frame()
            if frame is None:
                if _poll_quit_key():
                    break
                continue
            if result is not None:
                print(
                    f"EAR={result['ear']:.3f}  "
                    f"yaw={result['yaw']:+.1f}°  "
                    f"pitch={result['pitch']:+.1f}°  "
                    f"roll={result['roll']:+.1f}°",
                    end="\r",
                )
            else:
                print("No face detected.                    ", end="\r")
            cv2.imshow("AI Focus Tracker - Phase 1", frame)
            if _poll_quit_key():
                break
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.release()
        cv2.destroyAllWindows()
        print("\nStopped.", file=sys.stderr)
