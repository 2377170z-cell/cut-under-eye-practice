"""Color-correct an image or video and extract lower-eyelid ROIs.

Every input must contain a 24-patch Macbeth ColorChecker. For a video, the
model is searched for near the start, then reused for every frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np


# These are the lower eyelid arcs in MediaPipe's 478-landmark face mesh.
# "left" and "right" mean the person's left/right (not the viewer's).
LOWER_LID = {
    "left": [263, 249, 390, 373, 374, 380, 381, 382, 362],
    "right": [33, 7, 163, 144, 145, 153, 154, 155, 133],
}
UPPER_LID = {
    "left": [263, 466, 388, 387, 386, 385, 384, 398, 362],
    "right": [33, 246, 161, 160, 159, 158, 157, 173, 133],
}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_color_model(frame: np.ndarray):
    """Build a CCM from one BGR frame, following the supplied image example."""
    if not hasattr(cv2, "mcc") or not hasattr(cv2, "ccm"):
        raise RuntimeError(
            "cv2.mcc / cv2.ccm がありません。opencv-python を削除して、"
            "opencv-contrib-python をインストールしてください。"
        )

    detector = cv2.mcc.CCheckerDetector_create()
    detector.process(frame, cv2.mcc.MCC24)
    checker = detector.getBestColorChecker()
    if checker is None:
        return None

    # C++内部のクラッシュ（Unknown C++ exception）を防ぐための例外処理を追加
    try:
        charts_rgb = checker.getChartsRGB()
        src = charts_rgb[:, 1].copy().reshape(24, 1, 3).astype(np.float64) / 255.0
        model = cv2.ccm_ColorCorrectionModel(src, cv2.ccm.COLORCHECKER_Macbeth)
        model.setColorSpace(cv2.ccm.COLOR_SPACE_sRGB)
        model.run()
        return model
    except cv2.error as e:
        print(f"警告: カラーチェッカーの解析中にOpenCVエラーが発生しました（誤検出の可能性があります）: {e}")
        return None


def apply_color_model(frame_bgr: np.ndarray, model) -> np.ndarray:
    """Apply the CCM to a BGR uint8 frame and return a BGR uint8 frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    corrected = model.infer(rgb.astype(np.float64) / 255.0)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)


def find_color_model(cap: cv2.VideoCapture, max_frames: int):
    """Search the beginning of the video until a ColorChecker is found."""
    for frame_index in range(max_frames):
        ok, frame = cap.read()
        if not ok:
            break
        model = build_color_model(frame)
        if model is not None:
            print(f"ColorChecker detected in frame {frame_index}.")
            return model
    return None


def landmarks_to_pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float32)


def lower_eyelid_quad(
    points: np.ndarray,
    eye: str,
    top_offset: float,
    height_factor: float,
) -> np.ndarray:
    """Return a perspective-warp quadrilateral that covers the exposed lower lid.

    The top edge follows the lower lid. Its direction is derived from the upper
    to lower eyelid vector, so the crop follows modest head roll and tilt.
    """
    lower = points[LOWER_LID[eye]]
    upper = points[UPPER_LID[eye]]
    corner_a, corner_b = lower[0], lower[-1]
    lower_mid = lower[1:-1].mean(axis=0)
    upper_mid = upper[1:-1].mean(axis=0)

    down = lower_mid - upper_mid
    norm = float(np.linalg.norm(down))
    if norm < 1e-6:
        down = np.array([0.0, 1.0], dtype=np.float32)
    else:
        down /= norm
    # A lower lid must point down in the image; this prevents a flipped crop.
    if down[1] < 0:
        down *= -1.0

    eye_width = float(np.linalg.norm(corner_b - corner_a))
    top = top_offset * eye_width * down
    bottom = (top_offset + height_factor) * eye_width * down
    return np.float32([corner_a + top, corner_b + top, corner_b + bottom, corner_a + bottom])


def warp_roi(frame: np.ndarray, quad: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    destination = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(
        frame, transform, output_size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def selected_eyes(choice: str) -> Iterable[str]:
    return ("left", "right") if choice == "both" else (choice,)


def make_landmarker(model_path: Path, running_mode):
    if not model_path.is_file():
        raise FileNotFoundError(f"Face Landmarker model が見つかりません: {model_path}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def process_video(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(f"入力動画が見つかりません: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("入力動画を開けませんでした。")
    color_model = find_color_model(cap, args.calibration_search_frames)
    cap.release()
    if color_model is None:
        raise RuntimeError(
            "ColorChecker を検出できませんでした。動画の先頭に24色Macbeth "
            "ColorCheckerが鮮明に映るようにするか、--calibration-search-frames を増やしてください。"
        )

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("動画のフレームサイズを取得できませんでした。")

    full_path = Path(args.output_full)
    roi_path = Path(args.output_roi)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    full_writer = cv2.VideoWriter(str(full_path), fourcc, fps, (frame_width, frame_height))
    roi_width, roi_height = args.roi_width, args.roi_height
    roi_count = 2 if args.eye == "both" else 1
    roi_writer = cv2.VideoWriter(str(roi_path), fourcc, fps, (roi_width * roi_count, roi_height))
    if not full_writer.isOpened() or not roi_writer.isOpened():
        raise RuntimeError("出力動画を作成できませんでした。--codec を変更してみてください（例: mp4v）。")

    black_roi = np.zeros((roi_height, roi_width, 3), dtype=np.uint8)
    frame_index = 0
    try:
        with make_landmarker(Path(args.model), mp.tasks.vision.RunningMode.VIDEO) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                corrected = apply_color_model(frame, color_model)
                rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                rois = []
                if result.face_landmarks:
                    points = landmarks_to_pixels(result.face_landmarks[0], frame_width, frame_height)
                    for eye in selected_eyes(args.eye):
                        quad = lower_eyelid_quad(points, eye, args.roi_top_offset, args.roi_height_factor)
                        rois.append(warp_roi(corrected, quad, (roi_width, roi_height)))
                        cv2.polylines(corrected, [np.round(quad).astype(np.int32)], True, (0, 255, 0), 2)
                        label_position = tuple(np.round(quad[0]).astype(int))
                        cv2.putText(corrected, eye, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                else:
                    rois = [black_roi.copy() for _ in selected_eyes(args.eye)]
                    cv2.putText(corrected, "face not detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                full_writer.write(corrected)
                roi_writer.write(np.hstack(rois))
                frame_index += 1
                if frame_index % 100 == 0:
                    print(f"Processed {frame_index} frames")
    finally:
        cap.release()
        full_writer.release()
        roi_writer.release()

    print(f"Completed: {full_path}")
    print(f"Completed: {roi_path}")


def process_image(args: argparse.Namespace) -> None:
    """Process one photograph with the same CCM and lower-eyelid extraction."""
    input_path = Path(args.input)
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"画像を読み込めませんでした: {input_path}")

    color_model = build_color_model(frame)
    if color_model is None:
        raise RuntimeError("画像内から24色Macbeth ColorCheckerを検出できませんでした。")

    corrected = apply_color_model(frame, color_model)
    height, width = corrected.shape[:2]
    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with make_landmarker(Path(args.model), mp.tasks.vision.RunningMode.IMAGE) as landmarker:
        result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        raise RuntimeError("画像から顔を検出できませんでした。")

    points = landmarks_to_pixels(result.face_landmarks[0], width, height)
    rois = []
    for eye in selected_eyes(args.eye):
        quad = lower_eyelid_quad(points, eye, args.roi_top_offset, args.roi_height_factor)
        rois.append(warp_roi(corrected, quad, (args.roi_width, args.roi_height)))
        cv2.polylines(corrected, [np.round(quad).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(
            corrected,
            eye,
            tuple(np.round(quad[0]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

    full_path = Path(args.output_full or "outputs/color_corrected_with_roi.png")
    roi_path = Path(args.output_roi or "outputs/lower_eyelid_roi.png")
    full_path.parent.mkdir(parents=True, exist_ok=True)
    roi_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(full_path), corrected) or not cv2.imwrite(str(roi_path), np.hstack(rois)):
        raise RuntimeError("出力画像を保存できませんでした。")
    print(f"Completed: {full_path}")
    print(f"Completed: {roi_path}")


def process_input(args: argparse.Namespace) -> None:
    requested_path = Path(args.input)
    script_folder_path = Path(__file__).resolve().parent / requested_path
    candidates = (requested_path,) if requested_path.is_absolute() else (requested_path, script_folder_path)
    input_path = next((path for path in candidates if path.is_file()), None)
    if input_path is None:
        raise FileNotFoundError(
            f"入力ファイルが見つかりません: {requested_path}\n"
            f"スクリプトと同じフォルダも確認しました: {script_folder_path}"
        )
    # Allow a relative filename to work even when this script is run elsewhere.
    args.input = str(input_path)
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        process_image(args)
    else:
        args.output_full = args.output_full or "outputs/color_corrected_with_roi.mp4"
        args.output_roi = args.output_roi or "outputs/lower_eyelid_roi.mp4"
        process_video(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="入力写真または動画ファイル")
    parser.add_argument("--model", default="face_landmarker.task", help="MediaPipe Face Landmarker .task")
    parser.add_argument("--output-full", help="色補正済みの全体出力ファイル")
    parser.add_argument("--output-roi", help="下眼瞼ROIの出力ファイル")
    parser.add_argument("--eye", choices=("left", "right", "both"), default="both")
    parser.add_argument("--calibration-search-frames", type=int, default=180)
    parser.add_argument("--roi-width", type=int, default=320)
    parser.add_argument("--roi-height", type=int, default=120)
    parser.add_argument("--roi-top-offset", type=float, default=0.10,
                        help="下眼瞼縁からの開始位置（眼幅に対する比率）")
    parser.add_argument("--roi-height-factor", type=float, default=0.35,
                        help="切り出し高さ（眼幅に対する比率）")
    parser.add_argument("--codec", default="mp4v", help="FourCC (default: mp4v)")
    return parser.parse_args()


if __name__ == "__main__":
    process_input(parse_args())
