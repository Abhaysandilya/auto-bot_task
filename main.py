import argparse
import os
import time
import uuid
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from controller import SimController
from vision import (
    VisionConfig,
    pick_goal_pixel,
    detect_obstacle_mask,
    compute_repulsive_vector,
    compute_goal_vector,
    combine_vectors,
    detect_robot_center,
    vector_to_step,
    distance_to_goal,
)
from analyzer import RunResult, append_run_result, compute_average_collisions, plot_speed_vs_avg_collisions


def run_single(corner: str, controller: SimController, cfg: VisionConfig, args: argparse.Namespace) -> RunResult:
    run_id = str(uuid.uuid4())[:8]
    collisions = 0
    steps = 0
    obstacle_speed = float(args.obstacle_speed)

    # Capture first frame to set goal near chosen corner
    frame = controller.capture_frame()
    goal_xy = pick_goal_pixel(frame, corner, margin=args.goal_margin)

    last_distance = float("inf")
    stagnant_steps = 0
    prev_center: Optional[Tuple[int, int]] = None

    for step in range(args.steps_per_run):
        steps = step + 1
        frame = controller.capture_frame()
        center_xy = detect_robot_center(frame, previous_center=prev_center)
        prev_center = center_xy

        obstacle_mask = detect_obstacle_mask(frame, cfg)

        repel_vec = compute_repulsive_vector(obstacle_mask, center_xy)
        goal_vec = compute_goal_vector(center_xy, goal_xy)

        # Dynamic weighting: increase avoidance if obstacle density near center is high
        cx, cy = center_xy
        h, w = obstacle_mask.shape[:2]
        win = obstacle_mask[max(0, cy - 10): min(h, cy + 11), max(0, cx - 10): min(w, cx + 11)]
        density = float(np.mean(win) / 255.0) if win.size > 0 else 0.0
        w_goal = float(args.w_goal)
        w_avoid = float(args.w_avoid) * (1.0 + 2.0 * density)

        combined = combine_vectors(goal_vec, repel_vec, w_goal=w_goal, w_avoid=w_avoid)
        dx, dy = vector_to_step(combined, step_pixels=float(args.step_pixels), invert_y=cfg.invert_y_for_motion)

        resp = controller.move_relative(dx, dy, speed=None)
        if isinstance(resp, dict) and resp.get("collision") is True:
            collisions += 1

        # Evaluate progress
        next_frame = controller.capture_frame()
        next_center = detect_robot_center(next_frame, previous_center=center_xy)
        distance_now = distance_to_goal(next_center, goal_xy)
        prev_center = next_center

        if distance_now < args.distance_threshold:
            reached = True
            append_run_result(args.csv, RunResult(run_id, corner, steps, collisions, reached, obstacle_speed))
            return RunResult(run_id, corner, steps, collisions, reached, obstacle_speed)

        if distance_now >= last_distance - 1.0:
            stagnant_steps += 1
            if stagnant_steps % 12 == 0 and density > 0.12:
                collisions += 1
        else:
            stagnant_steps = 0
        last_distance = distance_now

        if args.save_frames:
            frames_dir = os.path.join(args.frames_dir, f"{run_id}_{corner}")
            os.makedirs(frames_dir, exist_ok=True)
            cv2_path = os.path.join(frames_dir, f"frame_{step:05d}.png")
            import cv2
            cv2.imwrite(cv2_path, frame)

        time.sleep(args.sleep)

    reached = False
    append_run_result(args.csv, RunResult(run_id, corner, steps, collisions, reached, obstacle_speed))
    return RunResult(run_id, corner, steps, collisions, reached, obstacle_speed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous collision-avoidance navigator for Sim-1")
    parser.add_argument("--base-url", type=str, default=os.getenv("SIM_BASE_URL", "http://localhost:5000"))
    parser.add_argument("--steps-per-run", type=int, default=500)
    parser.add_argument("--step-pixels", type=float, default=15.0)
    parser.add_argument("--goal-margin", type=int, default=40)
    parser.add_argument("--distance-threshold", type=float, default=35.0)
    parser.add_argument("--w-goal", type=float, default=1.5)
    parser.add_argument("--w-avoid", type=float, default=1.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--moving-obstacles", action="store_true")
    parser.add_argument("--obstacle-speed", type=float, default=0.0)
    parser.add_argument("--csv", type=str, default="results.csv")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir", type=str, default="runs_frames")
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs per corner")
    parser.add_argument("--corners", nargs="*", default=["tl", "tr", "br", "bl"], help="Corners order")
    parser.add_argument("--plot-speed-curve", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Run against mock simulator if real server unavailable")

    args = parser.parse_args()

    use_mock = args.mock or bool(int(os.getenv("SIM_USE_MOCK", "0")))
    controller = SimController(base_url=args.base_url, use_mock=use_mock)
    cfg = VisionConfig(goal_margin_pixels=args.goal_margin)

    if args.moving_obstacles:
        controller.set_moving_obstacles(True, speed=args.obstacle_speed)

    if args.plot_speed_curve:
        plot_speed_vs_avg_collisions(args.csv, "obstacle_speed_vs_collisions.png")
        print("Saved plot to obstacle_speed_vs_collisions.png")
        return

    all_results: List[RunResult] = []
    for corner in args.corners:
        for _ in range(max(1, args.repeat)):
            res = run_single(corner, controller, cfg, args)
            all_results.append(res)
            print(f"Run {res.run_id} corner={corner} steps={res.steps} collisions={res.collisions} reached={res.reached}")

    averages = compute_average_collisions(args.csv)
    if averages:
        print("Average collisions by corner:")
        for corner, avg in averages.items():
            print(f"  {corner}: {avg:.2f}")


if __name__ == "__main__":
    main()
