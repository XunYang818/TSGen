"""Navigation path visualization test script."""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yaml_layout.draw import House
from utils.navigation import Navigation


def visualize(yaml_path: str, start: tuple, end: tuple, out_dir: str, fmt: str = 'pdf'):
    """Draw navigation path and save figure."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6), dpi=400)

    margin = 10
    for ax in [ax_left, ax_right]:
        ax.set_xlim(-margin, 100 + margin)
        ax.set_ylim(-margin, 80 + margin)
        ax.set_aspect('equal')

    # Left: potential field + path
    nav = Navigation(yaml_path)
    path = nav.find_path(start, end, potential=True, potential_k=3,
                         potential_decay=1, potential_rand=True, smooth=True)
    potential = np.array(nav.get_potential())
    nav.draw_potential(ax_left, potential, cmap='viridis')
    ax_left.scatter(np.array(path)[:, 0], np.array(path)[:, 1], c='#f4755c', s=2)

    # Right: house layout + human-like trajectory
    house = House(ax_right, yaml_path)
    house.draw()
    trajectory = nav.generate_human_trajectory(
        path, speed=(0.8, 1), dt=1, start_time="2025-01-01 00:00:00",
        speed_start=0.5, speed_end=0.5, unit='dm'
    )
    nav.draw_trajectory(ax_right, trajectory, cmap='Blues', gap=1, linewidth=3, arrow_size=15, node_size=20)

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(yaml_path))[0]
    out_path = os.path.join(out_dir, f"{name}_nav.{fmt}")
    fig.savefig(out_path, format=fmt, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Navigation path visualization")
    parser.add_argument('-y', '--yaml', required=True, help='Layout YAML file path')
    parser.add_argument('-p', '--points', required=True,
                        help='Start and end coordinates, format: x1,y1,x2,y2')
    parser.add_argument('-o', '--output', default='./figures', help='Output directory (default: ./figures)')
    parser.add_argument('-f', '--format', default='pdf', choices=['pdf', 'svg'], help='Output format (default: pdf)')
    args = parser.parse_args()

    coords = list(map(float, args.points.split(',')))
    start = (coords[0], coords[1])
    end = (coords[2], coords[3])

    visualize(args.yaml, start, end, args.output, args.format)


if __name__ == '__main__':
    main()
