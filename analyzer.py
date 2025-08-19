import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt  # type: ignore


@dataclass
class RunResult:
    run_id: str
    corner: str
    steps: int
    collisions: int
    reached: bool
    obstacle_speed: float


def append_run_result(csv_path: str, result: RunResult) -> None:
    header = ["run_id", "corner", "steps", "collisions", "reached", "obstacle_speed"]
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": result.run_id,
                "corner": result.corner,
                "steps": result.steps,
                "collisions": result.collisions,
                "reached": int(result.reached),
                "obstacle_speed": result.obstacle_speed,
            }
        )


def compute_average_collisions(csv_path: str) -> Dict[str, float]:
    totals: Dict[str, Tuple[int, int]] = {}  # corner -> (sum_collisions, count)
    if not os.path.exists(csv_path):
        return {}
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corner = row.get("corner", "").lower()
            collisions = int(row.get("collisions", 0))
            if corner not in totals:
                totals[corner] = (collisions, 1)
            else:
                s, c = totals[corner]
                totals[corner] = (s + collisions, c + 1)
    return {corner: (s / c if c > 0 else 0.0) for corner, (s, c) in totals.items()}


def plot_speed_vs_avg_collisions(csv_path: str, output_png: str) -> None:
    # Aggregate by obstacle_speed
    speed_to_collisions: Dict[float, List[int]] = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed = float(row.get("obstacle_speed", 0.0))
            collisions = int(row.get("collisions", 0))
            speed_to_collisions.setdefault(speed, []).append(collisions)

    speeds = sorted(speed_to_collisions.keys())
    avg_collisions = [
        (sum(speed_to_collisions[s]) / max(1, len(speed_to_collisions[s]))) for s in speeds
    ]

    plt.figure(figsize=(7, 4))
    plt.plot(speeds, avg_collisions, marker="o")
    plt.xlabel("Obstacle Speed")
    plt.ylabel("Average Collisions")
    plt.title("Obstacle Speed vs Average Collisions")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close()
