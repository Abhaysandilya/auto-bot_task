import os
import time
from typing import List, Tuple

import numpy as np  # type: ignore
import streamlit as st  # type: ignore
import plotly.graph_objects as go  # type: ignore

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


st.set_page_config(page_title="Sim-1 Autonomous Runner", layout="wide")

st.title("Sim-1 Autonomous Runner")

with st.sidebar:
    st.header("Simulator Settings")
    base_url = st.text_input("Base URL", value=os.getenv("SIM_BASE_URL", "http://localhost:5000"))
    use_mock = st.checkbox("Use Mock Simulator", value=False)

    st.header("Run Parameters")
    steps_per_run = st.number_input("Steps per run", 100, 10000, 1500, step=100)
    step_pixels = st.slider("Step pixels", 1, 30, 10)
    goal_margin = st.slider("Goal margin (px)", 5, 200, 40)
    distance_threshold = st.slider("Distance threshold (px)", 10, 800, 250)
    w_goal = st.slider("Weight: Goal", 0.1, 5.0, 2.0, step=0.1)
    w_avoid = st.slider("Weight: Avoid", 0.1, 5.0, 1.0, step=0.1)
    sleep_s = st.slider("Sleep per step (s)", 0.0, 0.2, 0.01, step=0.005)

    st.header("Obstacles")
    moving = st.checkbox("Enable moving obstacles", value=False)
    obstacle_speed = st.slider("Obstacle speed", 0.0, 5.0, 1.0, step=0.1)

    st.header("Run Plan")
    corners = st.multiselect("Corners", ["tl", "tr", "br", "bl"], default=["tl", "tr", "br", "bl"])
    repeat = st.number_input("Repeat per corner", 1, 10, 1)

    st.header("Outputs")
    save_frames = st.checkbox("Save frames", value=True)
    frames_dir = st.text_input("Frames directory", value="runs_frames_ui")
    csv_path = st.text_input("Results CSV", value="results.csv")

run_btn = st.button("Run Experiment")
preview_btn = st.button("Quick 3D Preview")

# Tabs for 2D/3D
tab2d, tab3d = st.tabs(["2D Vision", "3D Viewer"]) 
frame_placeholder = tab2d.empty()
status_area = tab2d.empty()

# 3D Viewer placeholder
plot_placeholder = tab3d.empty()


def extract_obstacle_rects(obstacle_mask: np.ndarray, max_rects: int = 15) -> List[Tuple[int, int, int, int]]:
    import cv2
    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours]
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects[:max_rects]


def to_scene_coords(px: int, py: int, canvas_size: Tuple[int, int]) -> Tuple[float, float]:
    w_img, h_img = canvas_size
    x = (px / max(1, w_img) - 0.5) * 10.0
    y = (py / max(1, h_img) - 0.5) * 10.0
    return float(x), float(-y)


def plot3d(center_xy: Tuple[int, int], goal_xy: Tuple[int, int], rects: List[Tuple[int, int, int, int]], canvas_size: Tuple[int, int]):
    w_img, h_img = canvas_size
    rx, ry = to_scene_coords(center_xy[0], center_xy[1], canvas_size)
    gx, gy = to_scene_coords(goal_xy[0], goal_xy[1], canvas_size)

    fig = go.Figure()

    # Ground
    fig.add_trace(go.Surface(z=[[0,0],[0,0]], x=[-7,7], y=[-7,7], showscale=False, colorscale=[[0,'#BDBDBD'],[1,'#BDBDBD']], opacity=1.0))

    # Obstacles as boxes
    for (x, y, bw, bh) in rects:
        cx, cy = to_scene_coords(x + bw // 2, y + bh // 2, canvas_size)
        sx = max(0.35, bw / max(1, w_img) * 6.0)
        sy = max(0.35, bh / max(1, h_img) * 6.0)
        # Create a box via mesh3d
        X = [
            cx - sx/2, cx + sx/2, cx + sx/2, cx - sx/2,  # bottom square
            cx - sx/2, cx + sx/2, cx + sx/2, cx - sx/2   # top square
        ]
        Y = [
            cy - sy/2, cy - sy/2, cy + sy/2, cy + sy/2,
            cy - sy/2, cy - sy/2, cy + sy/2, cy + sy/2
        ]
        Z = [0,0,0,0, 1.0,1.0,1.0,1.0]
        I = [0,1,2, 0,2,3, 4,5,6, 4,6,7, 0,1,5, 0,5,4, 2,3,7, 2,7,6, 1,2,6, 1,6,5, 0,3,7, 0,7,4]
        fig.add_trace(go.Mesh3d(x=X, y=Y, z=Z, i=I, color='#39FF14', opacity=1.0))

    # Goal sphere
    fig.add_trace(go.Scatter3d(x=[gx], y=[gy], z=[0.33], mode='markers', marker=dict(size=8, color='#00E676')))

    # Robot body (cylinder approximated as a short line with size)
    fig.add_trace(go.Scatter3d(x=[rx], y=[ry], z=[0.45], mode='markers', marker=dict(size=10, color='#C62828')))
    # Robot head
    fig.add_trace(go.Scatter3d(x=[rx], y=[ry], z=[0.9], mode='markers', marker=dict(size=7, color='#FFFFFF')))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6,y=1.8,z=1.4))
        ),
        margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor='#EAEAEA'
    )

    plot_placeholder.plotly_chart(fig, use_container_width=True)


if preview_btn:
    controller = SimController(base_url=base_url, use_mock=True)
    frame = controller.capture_frame()
    cfg = VisionConfig(goal_margin_pixels=goal_margin)
    goal_xy = pick_goal_pixel(frame, "tr", margin=goal_margin)
    center_xy = detect_robot_center(frame)
    mask = detect_obstacle_mask(frame, cfg)
    h, w = mask.shape[:2]
    rects = extract_obstacle_rects(mask)
    plot3d(center_xy, goal_xy, rects, (w, h))

if run_btn:
    controller = SimController(base_url=base_url, use_mock=use_mock)
    cfg = VisionConfig(goal_margin_pixels=goal_margin)
    if moving:
        controller.set_moving_obstacles(True, speed=obstacle_speed)

    for corner in corners:
        for r in range(int(repeat)):
            run_id = f"ui_{int(time.time())}_{corner}_{r}"
            collisions = 0
            steps = 0

            frame = controller.capture_frame()
            goal_xy = pick_goal_pixel(frame, corner, margin=goal_margin)
            prev_center = None
            last_distance = float("inf")
            stagnant_steps = 0

            frames_out_dir = os.path.join(frames_dir, f"{run_id}")
            if save_frames:
                os.makedirs(frames_out_dir, exist_ok=True)

            for step in range(int(steps_per_run)):
                steps = step + 1
                frame = controller.capture_frame()
                center_xy = detect_robot_center(frame, previous_center=prev_center)
                prev_center = center_xy

                obstacle_mask = detect_obstacle_mask(frame, cfg)
                repel_vec = compute_repulsive_vector(obstacle_mask, center_xy)
                goal_vec = compute_goal_vector(center_xy, goal_xy)
                h, w = obstacle_mask.shape[:2]
                cx, cy = center_xy
                win = obstacle_mask[max(0, cy - 10): min(h, cy + 11), max(0, cx - 10): min(w, cx + 11)]
                density = float(np.mean(win) / 255.0) if win.size > 0 else 0.0
                combined = combine_vectors(goal_vec, repel_vec, w_goal=w_goal, w_avoid=w_avoid * (1 + 2 * density))
                dx, dy = vector_to_step(combined, step_pixels=float(step_pixels), invert_y=cfg.invert_y_for_motion)

                # 2D Updates
                import cv2
                frame_vis = frame.copy()
                cv2.circle(frame_vis, goal_xy, 6, (0, 255, 0), -1)
                cv2.circle(frame_vis, center_xy, 6, (0, 0, 255), -1)
                frame_placeholder.image(frame_vis[:, :, ::-1], caption=f"Run {run_id} step {steps}", use_container_width=True)

                # 3D Updates (throttled)
                if steps % 10 == 1:
                    rects = extract_obstacle_rects(obstacle_mask)
                    plot3d(center_xy, goal_xy, rects, (w, h))

                resp = controller.move_relative(dx, dy, speed=None)
                if isinstance(resp, dict) and resp.get("collision") is True:
                    collisions += 1

                next_frame = controller.capture_frame()
                next_center = detect_robot_center(next_frame, previous_center=center_xy)
                dist = distance_to_goal(next_center, goal_xy)
                prev_center = next_center

                status_area.write(f"Corner={corner} Steps={steps}/{steps_per_run} Collisions={collisions} Dist={dist:.1f}")

                if save_frames:
                    cv2.imwrite(os.path.join(frames_out_dir, f"frame_{step:05d}.png"), frame)

                if dist < float(distance_threshold):
                    append_run_result(csv_path, RunResult(run_id, corner, steps, collisions, True, float(obstacle_speed)))
                    break

                if dist >= last_distance - 1.0:
                    stagnant_steps += 1
                    if stagnant_steps % 12 == 0 and density > 0.12:
                        collisions += 1
                else:
                    stagnant_steps = 0
                last_distance = dist

                time.sleep(float(sleep_s))

            append_run_result(csv_path, RunResult(run_id, corner, steps, collisions, False, float(obstacle_speed)))

st.header("Results")
if os.path.exists("results.csv"):
    avgs = compute_average_collisions("results.csv")
    st.write({k: round(v, 2) for k, v in avgs.items()})
    if st.button("Plot Speed vs Collisions"):
        plot_speed_vs_avg_collisions("results.csv", "obstacle_speed_vs_collisions.png")
        st.image("obstacle_speed_vs_collisions.png")

st.caption("3D viewer now uses Plotly for reliability. If you still see issues, try another browser.")
