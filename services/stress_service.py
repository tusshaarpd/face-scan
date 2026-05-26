from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.image_utils import pil_to_cv

try:
    import mediapipe as mp
except Exception:
    mp = None

LEFT_EYE = [33, 133, 159, 145, 160, 144]
RIGHT_EYE = [263, 362, 386, 374, 385, 380]
LEFT_EYE_BOX = [33, 133, 159, 145]
RIGHT_EYE_BOX = [263, 362, 386, 374]
UNDER_EYE_LEFT = [130, 247, 30, 29, 27, 28, 56, 190]
UNDER_EYE_RIGHT = [359, 467, 260, 259, 257, 258, 286, 414]
FOREHEAD_PROXY = [10, 67, 297, 109, 338]
MOUTH_JAW = [61, 291, 13, 14, 199, 152, 172, 397]


def _distance(a, b) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _ear(points: list[tuple[float, float]], indices: list[int]) -> float:
    outer, inner, upper_a, lower_a, upper_b, lower_b = [points[i] for i in indices]
    vertical_a = _distance(upper_a, lower_a)
    vertical_b = _distance(upper_b, lower_b)
    horizontal = max(_distance(outer, inner), 1.0)
    return ((vertical_a + vertical_b) / 2) / horizontal


def _bbox(points: list[tuple[float, float]], indices: list[int], pad: int = 10) -> list[int]:
    xs = [points[i][0] for i in indices]
    ys = [points[i][1] for i in indices]
    return [int(min(xs) - pad), int(min(ys) - pad), int(max(xs) + pad), int(max(ys) + pad)]


def _roi_mean_luma(gray: np.ndarray, box: list[int]) -> float:
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.mean(gray[y1:y2, x1:x2]))


def _score_quality(gray: np.ndarray) -> tuple[float, float]:
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    lighting = float(np.mean(gray))
    blur_quality = max(0, min(100, blur_score / 2.4))
    lighting_quality = 100 - min(100, abs(lighting - 132) * 1.25)
    return round(blur_quality, 1), round(lighting_quality, 1)


def _haar_faces(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return []
    equalized = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        equalized,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(80, 80),
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def _fallback_face_observations(
    gray: np.ndarray,
    blur_quality: float,
    lighting_quality: float,
    faces: list[tuple[int, int, int, int]],
) -> dict[str, Any]:
    if len(faces) > 1:
        return {
            "face_count": len(faces),
            "landmark_quality": 25,
            "blur_quality": blur_quality,
            "lighting_quality": lighting_quality,
            "issues": ["Multiple faces detected. Use a single-person image for analysis."],
            "overlay": {
                "boxes": [
                    {"label": "Face", "coords": [x, y, x + w, y + h], "color": [248, 113, 113]}
                    for x, y, w, h in faces
                ],
                "landmarks": [],
                "heat_regions": [],
            },
            "zone_scores": {},
        }

    x, y, w, h = faces[0]
    face_luma = _roi_mean_luma(gray, [x, y, x + w, y + h])
    eye_band = [x + int(w * 0.15), y + int(h * 0.22), x + int(w * 0.85), y + int(h * 0.48)]
    under_eye_band = [x + int(w * 0.18), y + int(h * 0.42), x + int(w * 0.82), y + int(h * 0.62)]
    forehead_band = [x + int(w * 0.2), y + int(h * 0.08), x + int(w * 0.8), y + int(h * 0.28)]
    mouth_band = [x + int(w * 0.24), y + int(h * 0.62), x + int(w * 0.76), y + int(h * 0.88)]

    under_luma = _roi_mean_luma(gray, under_eye_band)
    forehead_luma = _roi_mean_luma(gray, forehead_band)
    under_eye_darkness = max(0, min(100, (face_luma - under_luma + 8) * 1.9))
    forehead_tension = max(0, min(100, (face_luma - forehead_luma + 6) * 1.4 + (100 - lighting_quality) * 0.15))
    eye_strain = max(0, min(100, under_eye_darkness * 0.45 + (100 - blur_quality) * 0.2 + (100 - lighting_quality) * 0.15))
    fatigue_score = max(0, min(100, eye_strain * 0.62 + under_eye_darkness * 0.25 + (100 - blur_quality) * 0.1))
    stress_score = max(0, min(100, forehead_tension * 0.35 + eye_strain * 0.35 + (100 - lighting_quality) * 0.15))
    wellness_score = max(0, min(100, 100 - (stress_score * 0.42 + fatigue_score * 0.44 + eye_strain * 0.14)))
    recovery_score = max(stress_score, fatigue_score * 0.95, eye_strain * 0.9)

    issues = [
        "Face detected with fallback detector, but precise landmarks were unavailable; confidence is reduced.",
    ]
    if blur_quality < 35:
        issues.append("Image appears blurry; move closer to the camera and hold still.")
    if lighting_quality < 35:
        issues.append("Lighting is uneven or too dim/bright; face a soft light source.")

    return {
        "face_count": 1,
        "landmark_quality": round(min(55, blur_quality * 0.25 + lighting_quality * 0.25 + 12), 1),
        "blur_quality": blur_quality,
        "lighting_quality": lighting_quality,
        "eye_openness": 0,
        "left_eye_openness": 0,
        "right_eye_openness": 0,
        "eye_asymmetry": 0,
        "under_eye_darkness": round(under_eye_darkness, 1),
        "forehead_tension": round(forehead_tension, 1),
        "jaw_tension": 0,
        "stress_score_local": round(stress_score, 1),
        "fatigue_score_local": round(fatigue_score, 1),
        "eye_strain_local": round(eye_strain, 1),
        "wellness_score_local": round(wellness_score, 1),
        "recovery_score_local": round(recovery_score, 1),
        "issues": issues,
        "zone_scores": {
            "eyes": round(eye_strain, 1),
            "under_eyes": round(under_eye_darkness, 1),
            "forehead": round(forehead_tension, 1),
            "mouth_jaw": 0,
        },
        "overlay": {
            "boxes": [
                {"label": "Face", "coords": [x, y, x + w, y + h], "color": [34, 197, 94]},
                {"label": "Eye band", "coords": eye_band, "color": [56, 189, 248]},
                {"label": "Under eye", "coords": under_eye_band, "color": [251, 191, 36]},
                {"label": "Forehead", "coords": forehead_band, "color": [248, 113, 113]},
                {"label": "Mouth/Jaw", "coords": mouth_band, "color": [34, 197, 94]},
            ],
            "landmarks": [],
            "heat_regions": [
                [x + w * 0.5, y + h * 0.22, w * 0.18, forehead_tension / 100],
                [x + w * 0.5, y + h * 0.38, w * 0.22, eye_strain / 100],
                [x + w * 0.5, y + h * 0.52, w * 0.2, under_eye_darkness / 100],
            ],
        },
    }


def analyze_face(image: Image.Image) -> dict[str, Any]:
    cv_image = pil_to_cv(image)
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    blur_quality, lighting_quality = _score_quality(gray)

    faces = []
    if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        try:
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.35,
            ) as face_mesh:
                result = face_mesh.process(rgb)
            faces = result.multi_face_landmarks or []
        except Exception:
            faces = []

    if not faces:
        fallback_faces = _haar_faces(gray)
        if fallback_faces:
            return _fallback_face_observations(gray, blur_quality, lighting_quality, fallback_faces)
        return {
            "face_count": 0,
            "landmark_quality": 0,
            "blur_quality": blur_quality,
            "lighting_quality": lighting_quality,
            "issues": ["No face detected. Center your face and try again."],
            "overlay": {"boxes": [], "landmarks": [], "heat_regions": []},
            "zone_scores": {},
        }

    if len(faces) > 1:
        return {
            "face_count": len(faces),
            "landmark_quality": 30,
            "blur_quality": blur_quality,
            "lighting_quality": lighting_quality,
            "issues": ["Multiple faces detected. Use a single-person image for analysis."],
            "overlay": {"boxes": [], "landmarks": [], "heat_regions": []},
            "zone_scores": {},
        }

    landmarks = faces[0].landmark
    points = [(lm.x * width, lm.y * height) for lm in landmarks]
    left_ear = _ear(points, LEFT_EYE)
    right_ear = _ear(points, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2
    eye_asymmetry = abs(left_ear - right_ear) / max(avg_ear, 0.01)
    eye_openness = max(0, min(100, (avg_ear - 0.14) / 0.18 * 100))

    left_eye_box = _bbox(points, LEFT_EYE_BOX, pad=14)
    right_eye_box = _bbox(points, RIGHT_EYE_BOX, pad=14)
    under_left_box = _bbox(points, UNDER_EYE_LEFT, pad=8)
    under_right_box = _bbox(points, UNDER_EYE_RIGHT, pad=8)
    forehead_box = _bbox(points, FOREHEAD_PROXY, pad=18)
    mouth_box = _bbox(points, MOUTH_JAW, pad=12)

    face_box = _bbox(points, list(range(0, min(len(points), 468))), pad=4)
    face_luma = _roi_mean_luma(gray, face_box)
    under_luma = (_roi_mean_luma(gray, under_left_box) + _roi_mean_luma(gray, under_right_box)) / 2
    under_eye_darkness = max(0, min(100, (face_luma - under_luma + 10) * 2.2))

    brow_gap = _distance(points[65], points[295]) / max(_distance(points[234], points[454]), 1)
    mouth_width = _distance(points[61], points[291])
    mouth_open = _distance(points[13], points[14]) / max(mouth_width, 1)
    jaw_tension = max(0, min(100, (0.12 - mouth_open) * 420 + eye_asymmetry * 35))
    forehead_tension = max(0, min(100, (0.42 - brow_gap) * 160 + under_eye_darkness * 0.18))
    eye_strain = max(0, min(100, (100 - eye_openness) * 0.58 + under_eye_darkness * 0.28 + eye_asymmetry * 65))
    fatigue_score = max(0, min(100, eye_strain * 0.58 + under_eye_darkness * 0.24 + (100 - blur_quality) * 0.09))
    stress_score = max(0, min(100, forehead_tension * 0.35 + jaw_tension * 0.3 + eye_strain * 0.25 + (100 - lighting_quality) * 0.1))
    wellness_score = max(0, min(100, 100 - (stress_score * 0.42 + fatigue_score * 0.44 + eye_strain * 0.14)))
    recovery_score = max(stress_score, fatigue_score * 0.95, eye_strain * 0.9)

    landmark_sample = [[round(points[i][0], 1), round(points[i][1], 1)] for i in range(0, min(468, len(points)), 12)]
    zone_scores = {
        "eyes": round(eye_strain, 1),
        "under_eyes": round(under_eye_darkness, 1),
        "forehead": round(forehead_tension, 1),
        "mouth_jaw": round(jaw_tension, 1),
    }
    issues = []
    if blur_quality < 35:
        issues.append("Image appears blurry; confidence is reduced.")
    if lighting_quality < 35:
        issues.append("Lighting is uneven or too dim/bright; confidence is reduced.")

    return {
        "face_count": 1,
        "landmark_quality": round(min(100, (blur_quality * 0.45 + lighting_quality * 0.35 + 20)), 1),
        "blur_quality": blur_quality,
        "lighting_quality": lighting_quality,
        "eye_openness": round(eye_openness, 1),
        "left_eye_openness": round(max(0, min(100, (left_ear - 0.14) / 0.18 * 100)), 1),
        "right_eye_openness": round(max(0, min(100, (right_ear - 0.14) / 0.18 * 100)), 1),
        "eye_asymmetry": round(eye_asymmetry * 100, 1),
        "under_eye_darkness": round(under_eye_darkness, 1),
        "forehead_tension": round(forehead_tension, 1),
        "jaw_tension": round(jaw_tension, 1),
        "stress_score_local": round(stress_score, 1),
        "fatigue_score_local": round(fatigue_score, 1),
        "eye_strain_local": round(eye_strain, 1),
        "wellness_score_local": round(wellness_score, 1),
        "recovery_score_local": round(recovery_score, 1),
        "issues": issues,
        "zone_scores": zone_scores,
        "overlay": {
            "boxes": [
                {"label": "Left eye", "coords": left_eye_box, "color": [56, 189, 248]},
                {"label": "Right eye", "coords": right_eye_box, "color": [56, 189, 248]},
                {"label": "Under eye", "coords": under_left_box, "color": [251, 191, 36]},
                {"label": "Under eye", "coords": under_right_box, "color": [251, 191, 36]},
                {"label": "Forehead", "coords": forehead_box, "color": [248, 113, 113]},
                {"label": "Mouth/Jaw", "coords": mouth_box, "color": [34, 197, 94]},
            ],
            "landmarks": landmark_sample,
            "heat_regions": [
                [(forehead_box[0] + forehead_box[2]) / 2, (forehead_box[1] + forehead_box[3]) / 2, 48, forehead_tension / 100],
                [(left_eye_box[0] + left_eye_box[2]) / 2, (left_eye_box[1] + left_eye_box[3]) / 2, 34, eye_strain / 100],
                [(right_eye_box[0] + right_eye_box[2]) / 2, (right_eye_box[1] + right_eye_box[3]) / 2, 34, eye_strain / 100],
                [(mouth_box[0] + mouth_box[2]) / 2, (mouth_box[1] + mouth_box[3]) / 2, 42, jaw_tension / 100],
            ],
        },
    }


def recovery_need(stress: float, fatigue: float, eye_strain: float) -> str:
    composite = max(stress, fatigue * 0.95, eye_strain * 0.9)
    if composite >= 68:
        return "High"
    if composite >= 38:
        return "Medium"
    return "Low"


def build_local_report(observations: dict[str, Any]) -> dict[str, Any]:
    if observations.get("face_count") != 1:
        return {
            "stress_score": 0,
            "fatigue_score": 0,
            "eye_strain": 0,
            "recovery_score": 0,
            "wellness_score": 0,
            "recovery_need": "Unknown",
            "wellness_summary": "A single clear face is required before wellness indicators can be estimated.",
            "recommendations": ["Retake the scan with one centered face and even lighting."],
            "confidence": 0.0,
            "contributing_factors": observations.get("issues", []),
            "limitations": ["No wellness estimate was generated because face validation did not pass."],
        }

    stress = float(observations.get("stress_score_local", 0))
    fatigue = float(observations.get("fatigue_score_local", 0))
    eye_strain = float(observations.get("eye_strain_local", 0))
    wellness = float(observations.get("wellness_score_local", 0))
    recovery = float(observations.get("recovery_score_local", max(stress, fatigue, eye_strain)))
    confidence = min(0.88, max(0.35, observations.get("landmark_quality", 0) / 100))
    need = recovery_need(stress, fatigue, eye_strain)

    recommendations = [
        "Take a short screen break and look at a distant object for 20 seconds.",
        "Drink water and relax your jaw, brow, and shoulders.",
        "Use softer lighting and reduce screen brightness if your eyes feel strained.",
    ]
    if need == "High":
        recommendations.insert(0, "Plan a 15-20 minute recovery break before returning to intense focus.")
    elif need == "Low":
        recommendations = ["Keep normal breaks in your routine and maintain steady hydration."] + recommendations[:1]

    factors = [
        f"Eye strain proxy: {eye_strain:.0f}/100",
        f"Under-eye darkness proxy: {observations.get('under_eye_darkness', 0):.0f}/100",
        f"Forehead tension proxy: {observations.get('forehead_tension', 0):.0f}/100",
        f"Jaw tension proxy: {observations.get('jaw_tension', 0):.0f}/100",
    ] + observations.get("issues", [])

    return {
        "stress_score": round(stress, 1),
        "fatigue_score": round(fatigue, 1),
        "eye_strain": round(eye_strain, 1),
        "recovery_score": round(recovery, 1),
        "wellness_score": round(wellness, 1),
        "recovery_need": need,
        "wellness_summary": (
            "Local computer vision found visual fatigue and tension indicators that suggest "
            f"a {need.lower()} recovery need. Interpret this as a wellness cue, not a diagnosis."
        ),
        "recommendations": recommendations,
        "confidence": round(confidence, 2),
        "contributing_factors": factors,
        "limitations": [
            "Scores are based on one image and simple visual proxies.",
            "Lighting, camera angle, skin tone, expression, and image quality can affect results.",
            "This system does not diagnose disease, mental illness, burnout, or medical conditions.",
        ],
    }
