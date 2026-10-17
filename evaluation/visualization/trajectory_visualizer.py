#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trajectory visualization tool.
Usage: python trajectory_visualizer.py -j <json_path> [-o <output_dir>]
"""

import sys
import os
import json
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from yaml_layout.draw import House
from utils.navigation import Navigation

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'BuGn', 'YlOrRd', 'RdPu']
MAX_SEMANTIC_DISPLAY = 20


def get_yaml_path(json_path):
    """Infer yaml path from json path: 1-Accountant.json -> yaml_layout/data/1.yaml"""
    json_name = Path(json_path).stem
    yaml_num = json_name.split('-')[0]
    script_dir = Path(__file__).parent
    yaml_path = script_dir.parent.parent / "yaml_layout" / "data" / f"{yaml_num}.yaml"
    return str(yaml_path)


def fmt_time(t):
    """Format time: keep only HH:MM:SS"""
    if not t:
        return ""
    try:
        if 'T' in t:
            return t.split('T')[1][:8]
        elif ' ' in t:
            return t.split(' ')[1][:8]
        return t[:8]
    except (IndexError, AttributeError):
        return t


class TrajectoryVisualizer:
    """Trajectory visualizer."""

    def __init__(self, json_path, yaml_path=None):
        self.json_path = json_path
        self.yaml_path = yaml_path or get_yaml_path(json_path)

        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.trajs = data.get('trajectories', [])
        self.plans = {p['activity_id']: p for p in data.get('activity_plans', [])}
        self.events = {e['event_id']: e for e in data.get('events', [])}
        self.steps = {s['action_step_id']: s for s in data.get('action_steps', [])}

        print(f"[Load] {len(self.trajs)} trajectories, yaml: {self.yaml_path}")

    def _get_hierarchy(self, traj):
        """Get trajectory hierarchy: activity -> event -> action_step"""
        step = self.steps.get(traj.get('action_step_id'))
        if not step:
            return None, None, None

        event = self.events.get(step.get('event_id'))
        activity = self.plans.get(event.get('activity_id')) if event else None
        return activity, event, step

    def _draw_traj(self, ax, traj, color_idx):
        """Draw single trajectory."""
        points = traj.get('points', [])
        if not points:
            return

        step = self.steps.get(traj.get('action_step_id'))
        is_stationary = step and step.get('action_type') == 'stationary'

        if is_stationary:
            n = min(3, len(points))
            if len(points) > 3:
                pts = [points[i] for i in sorted(random.sample(range(len(points)), n))]
            else:
                pts = points
            xs = [p['x'] for p in pts]
            ys = [p['y'] for p in pts]
            ax.scatter(xs, ys, s=40, c='white', edgecolors='black', linewidth=1, zorder=5)
        else:
            nav = Navigation(self.yaml_path)
            pts = [(p['x'], p['y'], p.get('timestamp', '')) for p in points]
            nav.draw_trajectory(ax, pts, cmap=COLORS[color_idx % len(COLORS)], gap=0.8)

    def _build_info(self, trajs, start_idx, max_display=MAX_SEMANTIC_DISPLAY):
        """Build info lines for display, returns [(full_line, mask_line), ...]"""
        lines = []
        lines.append((f"Total Trajectories: {len(trajs)}", f"Total Trajectories: {len(trajs)}"))

        display_trajs = trajs[:max_display]
        if len(trajs) > max_display:
            lines.append((f"(Showing first {max_display} trajectories)",
                          f"(Showing first {max_display} trajectories)"))

        cur_act_id, cur_evt_id = None, None

        max_name_len = 0
        for traj in display_trajs:
            _, _, step = self._get_hierarchy(traj)
            if step:
                max_name_len = max(max_name_len, len(step.get('action_step_name', '')))

        for i, traj in enumerate(display_trajs):
            idx = start_idx + i + 1
            activity, event, step = self._get_hierarchy(traj)

            act_id = activity.get('activity_id') if activity else None
            evt_id = event.get('event_id') if event else None

            new_activity = False
            if act_id != cur_act_id:
                cur_act_id, cur_evt_id = act_id, None
                new_activity = True
                if activity:
                    lines.append(("", ""))
                    name = activity.get('activity_name', '').upper()
                    t1, t2 = fmt_time(activity.get('start_time')), fmt_time(activity.get('end_time'))
                    time_part = f"({t1}-{t2})"
                    lines.append((f"| {name} {time_part}", f"| {name} {' ' * len(time_part)}"))
                    lines.append(("", ""))

            if evt_id != cur_evt_id:
                cur_evt_id = evt_id
                if event:
                    if not new_activity:
                        lines.append(("", ""))
                    name = event.get('event_name', '')
                    t1, t2 = fmt_time(event.get('start_time')), fmt_time(event.get('end_time'))
                    time_part = f"({t1}-{t2})"
                    lines.append((f"  >> {name} {time_part}", f"  >> {name} {' ' * len(time_part)}"))

            if step:
                name = step.get('action_step_name', '').ljust(max_name_len)
                t1, t2 = fmt_time(step.get('start_time')), fmt_time(step.get('end_time'))
                action_type = step.get('action_type', '')
                marker = {'movement': '->', 'stationary': '*', 'absent': 'o'}.get(action_type, '?')

                detail = f"{t1}-{t2}"
                if action_type == 'movement':
                    mv = step.get('movement_detail', {})
                    if mv:
                        s, e = mv.get('start_position', {}), mv.get('end_position', {})
                        detail += f" | ({s.get('x',0):.0f},{s.get('y',0):.0f})->({e.get('x',0):.0f},{e.get('y',0):.0f})"
                elif action_type == 'stationary':
                    st = step.get('stationary_detail', {})
                    if st:
                        c = st.get('center_position', {})
                        detail += f" | @({c.get('x',0):.0f},{c.get('y',0):.0f})"
                elif action_type == 'absent':
                    ab = step.get('absent_detail', {})
                    reason = ab.get('reason', 'Away') if ab else 'Away'
                    detail += f" | {reason}"

                detail_part = f"({detail})"
                prefix = f"     [{idx}]{marker} {name} "
                lines.append((f"{prefix}{detail_part}", f"{prefix}{' ' * len(detail_part)}"))

        return lines

    def plot_all(self, title="Trajectory"):
        """Plot all trajectories in a single figure."""
        if not self.trajs:
            return None

        total = len(self.trajs)
        h = max(8, min(20, (total - 3) * 0.8))
        fig = plt.figure(figsize=(18, h), dpi=130)
        gs = gridspec.GridSpec(1, 2, width_ratios=[6, 4], figure=fig, wspace=0.05)
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        ax1.set_xlim(-5, 120)
        ax1.set_ylim(-5, 120)
        ax1.set_aspect('equal')
        ax1.set_title(f'{title} - All {total} Trajectories', fontsize=16, fontweight='bold')

        House(ax1, self.yaml_path).draw()
        for j, traj in enumerate(self.trajs):
            self._draw_traj(ax1, traj, j)

        ax2.axis('off')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Activity Summary', fontsize=14, fontweight='bold')
        lines = self._build_info(self.trajs, 0)

        y = 0.97
        line_height = 0.028

        for full_line, mask_line in lines:
            ax2.text(0.02, y, full_line, transform=ax2.transAxes, fontsize=11,
                     va='top', ha='left', fontfamily='monospace', color='#888888')
            ax2.text(0.02, y, mask_line, transform=ax2.transAxes, fontsize=11,
                     va='top', ha='left', fontfamily='monospace', color='black')
            y -= line_height

        plt.tight_layout()
        return fig

    def generate(self, output_dir):
        """Generate trajectory visualization."""
        total = len(self.trajs)
        if total == 0:
            print("[Error] No trajectory data")
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        print(f"[Output] {out_path}, total trajectories: {total}")

        fig = self.plot_all(Path(self.json_path).stem)
        if fig:
            filename = f"{Path(self.json_path).stem}.svg"
            fig.savefig(out_path / filename, bbox_inches='tight', format='svg')
            plt.close(fig)
            print(f"[Saved] {filename}")

        print(f"[Done] Figure saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Trajectory visualization tool')
    parser.add_argument('-j', '--json', required=True, help='Trajectory JSON file path')
    parser.add_argument('-y', '--layout', default=None,
                        help='House layout YAML path (optional; inferred when omitted)')
    parser.add_argument('-o', '--output', default='./figures', help='Output directory')

    args = parser.parse_args()

    if not os.path.exists(args.json):
        print(f"[Error] JSON file not found: {args.json}")
        sys.exit(1)

    print("=" * 50)
    print(f"  JSON: {args.json}")
    print(f"  Output: {args.output}")
    print("=" * 50)

    TrajectoryVisualizer(args.json, args.layout).generate(args.output)


if __name__ == '__main__':
    main()
