from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore


@dataclass
class VisionConfig:
	goal_margin_pixels: int = 40
	canny_low_threshold: int = 60
	canny_high_threshold: int = 140
	morph_kernel_size: int = 3
	obstacle_dilate_iter: int = 1
	invert_y_for_motion: bool = False  # Some sims use y-up; invert to y-down by default


def pick_goal_pixel(frame: np.ndarray, corner: str, margin: int) -> Tuple[int, int]:
	h, w = frame.shape[:2]
	margin_x = max(0, min(margin, w // 4))
	margin_y = max(0, min(margin, h // 4))
	corner = corner.lower()
	if corner in ("tl", "top_left"):
		return margin_x, margin_y
	if corner in ("tr", "top_right"):
		return w - margin_x - 1, margin_y
	if corner in ("br", "bottom_right"):
		return w - margin_x - 1, h - margin_y - 1
	if corner in ("bl", "bottom_left"):
		return margin_x, h - margin_y - 1
	# Default
	return w - margin_x - 1, h - margin_y - 1


def detect_obstacle_mask(frame: np.ndarray, cfg: VisionConfig) -> np.ndarray:
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edges = cv2.Canny(blurred, cfg.canny_low_threshold, cfg.canny_high_threshold)
	# Morphology to thicken edges → obstacles
	kernel = np.ones((cfg.morph_kernel_size, cfg.morph_kernel_size), np.uint8)
	dilated = cv2.dilate(edges, kernel, iterations=cfg.obstacle_dilate_iter)
	obstacle_mask = cv2.threshold(dilated, 1, 255, cv2.THRESH_BINARY)[1]
	return obstacle_mask


def detect_robot_center(frame: np.ndarray, previous_center: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
	"""
	Heuristic robot detector:
	- Convert to HSV and select moderately saturated, non-white, non-black regions
	- Prefer the most circular, sizable contour
	- Fallback to previous_center or image center
	"""
	h, w = frame.shape[:2]
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	hch, sch, vch = cv2.split(hsv)

	# Threshold for "colored" regions (exclude whites/grays/blacks)
	sat_mask = cv2.inRange(sch, 60, 255)
	val_mask = cv2.inRange(vch, 60, 255)
	colored = cv2.bitwise_and(sat_mask, val_mask)

	# Morph cleanup
	kernel = np.ones((3, 3), np.uint8)
	colored = cv2.morphologyEx(colored, cv2.MORPH_OPEN, kernel, iterations=1)
	colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, kernel, iterations=1)

	# Find contours and select by circularity and area
	contours, _ = cv2.findContours(colored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	best_center: Optional[Tuple[int, int]] = None
	best_score = -1.0

	min_area = float(w * h) * 0.0005  # heuristic
	max_area = float(w * h) * 0.05

	for cnt in contours:
		area = float(cv2.contourArea(cnt))
		if area < min_area or area > max_area:
			continue
		perimeter = float(cv2.arcLength(cnt, True))
		if perimeter <= 1e-3:
			continue
		circularity = 4.0 * np.pi * area / max(1e-6, perimeter * perimeter)
		M = cv2.moments(cnt)
		if M["m00"] == 0:
			continue
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
		# Prefer more circular and closer to previous center if provided
		score = float(circularity)
		if previous_center is not None:
			px, py = previous_center
			dist_penalty = np.hypot(cx - px, cy - py)
			score -= 0.001 * dist_penalty
		if score > best_score:
			best_score = score
			best_center = (cx, cy)

	if best_center is not None:
		return best_center

	if previous_center is not None:
		return previous_center

	return w // 2, h // 2


def compute_repulsive_vector(obstacle_mask: np.ndarray, center_xy: Tuple[int, int]) -> Tuple[float, float]:
	# Free space is non-obstacle area
	free_mask = 255 - obstacle_mask
	free_mask = free_mask.astype(np.uint8)

	# Distance transform: distance to nearest obstacle for each pixel in free space
	dt = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)

	cx, cy = center_xy
	h, w = dt.shape[:2]
	cx = int(np.clip(cx, 0, w - 1))
	cy = int(np.clip(cy, 0, h - 1))

	# If center is inside obstacle (dt==0), nudge to nearest free pixel
	if dt[cy, cx] <= 1e-3:
		# Search in a small window for max dt
		window = dt[max(0, cy - 5): min(h, cy + 6), max(0, cx - 5): min(w, cx + 6)]
		if window.size > 0:
			max_idx = np.argmax(window)
			wy, wx = np.unravel_index(max_idx, window.shape)
			cy = int(np.clip(cy - 5 + wy, 0, h - 1))
			cx = int(np.clip(cx - 5 + wx, 0, w - 1))

	# Gradient ascent on distance map → direction away from obstacles
	grad_y, grad_x = np.gradient(dt)
	vx = float(grad_x[cy, cx])
	vy = float(grad_y[cy, cx])

	mag = float(np.hypot(vx, vy))
	if mag < 1e-6:
		return 0.0, 0.0
	return vx / mag, vy / mag


def compute_goal_vector(center_xy: Tuple[int, int], goal_xy: Tuple[int, int]) -> Tuple[float, float]:
	cx, cy = center_xy
	gx, gy = goal_xy
	vx = float(gx - cx)
	vy = float(gy - cy)
	mag = float(np.hypot(vx, vy))
	if mag < 1e-6:
		return 0.0, 0.0
	return vx / mag, vy / mag


def combine_vectors(goal_vec: Tuple[float, float], repel_vec: Tuple[float, float], w_goal: float, w_avoid: float) -> Tuple[float, float]:
	vx = w_goal * goal_vec[0] + w_avoid * repel_vec[0]
	vy = w_goal * goal_vec[1] + w_avoid * repel_vec[1]
	mag = float(np.hypot(vx, vy))
	if mag < 1e-6:
		return 0.0, 0.0
	return vx / mag, vy / mag


def estimate_robot_center(frame: np.ndarray) -> Tuple[int, int]:
	h, w = frame.shape[:2]
	return w // 2, h // 2


def vector_to_step(vec: Tuple[float, float], step_pixels: float, invert_y: bool) -> Tuple[float, float]:
	vx, vy = vec
	dy = -vy if invert_y else vy
	return float(vx * step_pixels), float(dy * step_pixels)


def distance_to_goal(center_xy: Tuple[int, int], goal_xy: Tuple[int, int]) -> float:
	cx, cy = center_xy
	gx, gy = goal_xy
	return float(np.hypot(gx - cx, gy - cy))
