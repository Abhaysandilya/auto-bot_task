import os
import time
import base64
from typing import Any, Dict, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore
import requests


class SimController:
    """
    Thin client for interacting with the Sim-1 simulator HTTP API.

    Supports a mock fallback for offline development and demo using
    env var SIM_USE_MOCK=1 or constructor flag use_mock=True.
    """

    def __init__(
        self,
        base_url: str,
        capture_endpoint: Optional[str] = None,
        move_rel_endpoint: Optional[str] = None,
        move_endpoint: Optional[str] = None,
        state_endpoint: Optional[str] = None,
        timeout_seconds: float = 5.0,
        use_mock: Optional[bool] = None,
        mock_frame_size: Tuple[int, int] = (640, 480),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.capture_endpoint = capture_endpoint or os.getenv("CAPTURE_ENDPOINT", "/capture")
        self.move_rel_endpoint = move_rel_endpoint or os.getenv("MOVE_REL_ENDPOINT", "/move_rel")
        self.move_endpoint = move_endpoint or os.getenv("MOVE_ENDPOINT", "/move")
        self.state_endpoint = state_endpoint or os.getenv("STATE_ENDPOINT", "/state")
        self.session = requests.Session()
        self.timeout_seconds = timeout_seconds
        self.use_mock = bool(int(os.getenv("SIM_USE_MOCK", "0"))) if use_mock is None else use_mock
        self.mock_w, self.mock_h = int(mock_frame_size[0]), int(mock_frame_size[1])
        self._mock_tick = 0

    # -----------------------------
    # Helpers
    # -----------------------------
    def _full_url(self, endpoint: str) -> str:
        return f"{self.base_url}{endpoint}"

    def _decode_image(self, resp: requests.Response) -> np.ndarray:
        content_type = resp.headers.get("Content-Type", "")
        if content_type.startswith("image/"):
            image_bytes = resp.content
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError("Failed to decode image bytes from /capture response")
            return frame
        # Try JSON with base64
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unexpected /capture response format: {exc}") from exc
        for key in ("image", "frame", "data"):
            if key in data:
                b64_str = data[key]
                image_bytes = base64.b64decode(b64_str)
                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError("Failed to decode base64 image from /capture JSON response")
                return frame
        raise RuntimeError("No image field found in /capture JSON response")

    def _mock_frame(self) -> np.ndarray:
        # Simple synthetic scene: white background with moving black rectangles (obstacles)
        w, h = self.mock_w, self.mock_h
        frame = np.full((h, w, 3), 255, dtype=np.uint8)
        t = self._mock_tick
        # Draw 3 moving obstacles
        for i in range(3):
            x = (50 + 150 * i + (t * (i + 1)) % (w - 100))
            y = (100 + 60 * i + (t * (i + 2)) % (h - 80))
            x1 = int(max(0, min(w - 1, x)))
            y1 = int(max(0, min(h - 1, y)))
            x2 = int(max(0, min(w - 1, x1 + 60)))
            y2 = int(max(0, min(h - 1, y1 + 30)))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        self._mock_tick += 5
        return frame

    # -----------------------------
    # Public API
    # -----------------------------
    def capture_frame(self) -> np.ndarray:
        if self.use_mock:
            return self._mock_frame()
        url = self._full_url(self.capture_endpoint)
        try:
            resp = self.session.get(url, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return self._decode_image(resp)
        except Exception:
            if bool(int(os.getenv("SIM_ALLOW_FALLBACK", "1"))):
                self.use_mock = True
                return self._mock_frame()
            raise

    def get_state(self) -> Optional[Dict[str, Any]]:
        if self.use_mock:
            return {"mock": True, "tick": self._mock_tick}
        try:
            url = self._full_url(self.state_endpoint)
            resp = self.session.get(url, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    def move_relative(self, dx: float, dy: float, speed: Optional[float] = None) -> Dict[str, Any]:
        if self.use_mock:
            # In mock mode, we simulate movement without real physics and no collision flag
            return {"ok": True, "collision": False, "dx": dx, "dy": dy}
        payload: Dict[str, Any] = {"dx": float(dx), "dy": float(dy)}
        if speed is not None:
            payload["speed"] = float(speed)
        # Attempt /move_rel first
        try:
            url = self._full_url(self.move_rel_endpoint)
            resp = self.session.post(url, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            return self._safe_json(resp)
        except Exception:
            # Fallback to /move with relative=true
            try:
                url = self._full_url(self.move_endpoint)
                payload_fallback = {"x": float(dx), "y": float(dy), "relative": True}
                if speed is not None:
                    payload_fallback["speed"] = float(speed)
                resp = self.session.post(url, json=payload_fallback, timeout=self.timeout_seconds)
                resp.raise_for_status()
                return self._safe_json(resp)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to move robot: {exc}") from exc

    def _safe_json(self, resp: requests.Response) -> Dict[str, Any]:
        try:
            return resp.json()
        except Exception:
            return {}

    def set_moving_obstacles(self, enabled: bool, speed: float = 0.0) -> bool:
        if self.use_mock:
            return True
        payload = {"enabled": bool(enabled), "speed": float(speed)}
        candidate_endpoints = [
            os.getenv("MOVING_OBSTACLES_ENDPOINT", "/moving_obstacles"),
            "/obstacles/move",
            "/set_moving_obstacles",
        ]
        for endpoint in candidate_endpoints:
            try:
                url = self._full_url(endpoint)
                resp = self.session.post(url, json=payload, timeout=self.timeout_seconds)
                if resp.status_code // 100 == 2:
                    return True
            except Exception:
                continue
        return False

    def wait(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_vector(vec: Tuple[float, float], eps: float = 1e-6) -> Tuple[float, float]:
    vx, vy = vec
    mag = float(np.hypot(vx, vy))
    if mag < eps:
        return 0.0, 0.0
    return vx / mag, vy / mag
