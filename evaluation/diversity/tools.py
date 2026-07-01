"""Utility functions for data loading and furniture sequence building."""
import json
import yaml
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Furniture:
    """Furniture information."""
    id: str
    name: str
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, px: float, py: float) -> bool:
        """Check if point is inside furniture."""
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height


@dataclass
class ActionStep:
    """Action step information."""
    action_type: str  # movement / stationary / absent
    start_pos: Optional[Tuple[float, float]]
    end_pos: Optional[Tuple[float, float]]
    center_pos: Optional[Tuple[float, float]]


def load_json(path: str) -> dict:
    """Load JSON trajectory data."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(path: str) -> dict:
    """Load YAML layout data."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_furniture_list(yaml_data: dict) -> List[Furniture]:
    """Extract furniture list from YAML layout."""
    furniture_list = []
    house = yaml_data.get('house', {})
    rooms = house.get('rooms', [])

    for room in rooms:
        for furn in room.get('furniture', []):
            pos = furn.get('position', {})
            furniture_list.append(Furniture(
                id=furn.get('id', ''),
                name=furn.get('name', ''),
                x=pos.get('x', 0),
                y=pos.get('y', 0),
                width=pos.get('width', 0),
                height=pos.get('height', 0)
            ))
    return furniture_list


def extract_action_steps(json_data: dict) -> List[ActionStep]:
    """Extract action steps from JSON data."""
    steps = []
    for step in json_data.get('action_steps', []):
        action_type = step.get('action_type', '')
        start_pos = end_pos = center_pos = None

        if action_type == 'movement':
            md = step.get('movement_detail', {})
            if md:
                sp = md.get('start_position', {})
                ep = md.get('end_position', {})
                start_pos = (sp.get('x'), sp.get('y')) if sp else None
                end_pos = (ep.get('x'), ep.get('y')) if ep else None
        elif action_type == 'stationary':
            sd = step.get('stationary_detail', {})
            if sd:
                cp = sd.get('center_position', {})
                center_pos = (cp.get('x'), cp.get('y')) if cp else None

        steps.append(ActionStep(
            action_type=action_type,
            start_pos=start_pos,
            end_pos=end_pos,
            center_pos=center_pos
        ))
    return steps


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_furniture_at_point(px: float, py: float, furniture_list: List[Furniture]) -> str:
    """
    Find furniture ID at given point.
    - If point is inside a furniture, return that furniture ID
    - Otherwise return the nearest furniture ID
    """
    for furn in furniture_list:
        if furn.contains(px, py):
            return furn.id

    min_dist = float('inf')
    nearest_id = ''
    for furn in furniture_list:
        d = distance((px, py), furn.center)
        if d < min_dist:
            min_dist = d
            nearest_id = furn.id
    return nearest_id


def merge_consecutive_steps(action_steps: List[ActionStep]) -> List[ActionStep]:
    """
    Merge consecutive same-type trajectory segments.
    - Consecutive movements: merge into one, use first start and last end
    - Consecutive stationary: merge into one, use first center
    - Absent: ignored
    """
    if not action_steps:
        return []

    merged = []
    i = 0
    while i < len(action_steps):
        current = action_steps[i]

        if current.action_type == 'absent':
            i += 1
            continue

        if current.action_type == 'movement':
            start_pos = current.start_pos
            end_pos = current.end_pos
            j = i + 1
            while j < len(action_steps) and action_steps[j].action_type == 'movement':
                end_pos = action_steps[j].end_pos
                j += 1
            merged.append(ActionStep(
                action_type='movement',
                start_pos=start_pos,
                end_pos=end_pos,
                center_pos=None
            ))
            i = j
        elif current.action_type == 'stationary':
            center_pos = current.center_pos
            j = i + 1
            while j < len(action_steps) and action_steps[j].action_type == 'stationary':
                j += 1
            merged.append(ActionStep(
                action_type='stationary',
                start_pos=None,
                end_pos=None,
                center_pos=center_pos
            ))
            i = j
        else:
            i += 1

    return merged


def build_furniture_sequence(json_data: dict, yaml_data: dict, merge: bool = False) -> List[str]:
    """
    Build furniture interaction sequence.
    - Movement: extract start and end positions, get two furniture IDs
    - Stationary: extract center position, get one furniture ID
    - Absent: ignored

    Args:
        json_data: JSON trajectory data
        yaml_data: YAML layout data
        merge: Whether to merge consecutive same-type segments
    """
    furniture_list = extract_furniture_list(yaml_data)
    action_steps = extract_action_steps(json_data)

    if merge:
        action_steps = merge_consecutive_steps(action_steps)

    sequence = []
    for step in action_steps:
        if step.action_type == 'movement':
            if step.start_pos and step.start_pos[0] is not None:
                furn_id = find_furniture_at_point(step.start_pos[0], step.start_pos[1], furniture_list)
                if furn_id:
                    sequence.append(furn_id)
            if step.end_pos and step.end_pos[0] is not None:
                furn_id = find_furniture_at_point(step.end_pos[0], step.end_pos[1], furniture_list)
                if furn_id:
                    sequence.append(furn_id)
        elif step.action_type == 'stationary':
            if step.center_pos and step.center_pos[0] is not None:
                furn_id = find_furniture_at_point(step.center_pos[0], step.center_pos[1], furniture_list)
                if furn_id:
                    sequence.append(furn_id)

    return sequence


def extract_trajectory_points(json_data: dict) -> List[Tuple[float, float]]:
    """Extract all trajectory point coordinates (for spatial distribution calculation)."""
    points = []
    for traj in json_data.get('trajectories', []):
        for pt in traj.get('points', []):
            x, y = pt.get('x'), pt.get('y')
            if x is not None and y is not None:
                points.append((x, y))
    return points


def extract_trajectory_points_from_actions(json_data: dict) -> List[Tuple[float, float]]:
    """
    Extract position points from action_steps (when trajectories is empty).
    Includes all movement start/end points and stationary center points.
    """
    points = []
    action_steps = extract_action_steps(json_data)

    for step in action_steps:
        if step.action_type == 'movement':
            if step.start_pos and step.start_pos[0] is not None:
                points.append(step.start_pos)
            if step.end_pos and step.end_pos[0] is not None:
                points.append(step.end_pos)
        elif step.action_type == 'stationary':
            if step.center_pos and step.center_pos[0] is not None:
                points.append(step.center_pos)
    return points


def get_trajectory_points(json_data: dict) -> List[Tuple[float, float]]:
    """Get trajectory points, prefer from trajectories, fallback to action_steps."""
    points = extract_trajectory_points(json_data)
    if not points:
        points = extract_trajectory_points_from_actions(json_data)
    return points
