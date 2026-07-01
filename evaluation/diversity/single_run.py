"""Single run: compare diversity metrics between two trajectory files."""
import argparse
from typing import Dict, Union

from .tools import load_json, load_yaml, build_furniture_sequence, get_trajectory_points
from .metrics import compute_spatial_jsd, compute_ngram_jsd, compute_edit_distance


def compute_all_metrics(
    json1: dict,
    json2: dict,
    yaml_data: dict,
    merge: bool = False,
    grid_size: float = 3.0
) -> Dict[str, Union[int, float]]:
    """
    Compute all diversity metrics between two trajectories.

    Args:
        json1: JSON data for trajectory 1
        json2: JSON data for trajectory 2
        yaml_data: YAML layout data
        merge: Whether to merge consecutive same-type segments
        grid_size: Spatial grid size in decimeters

    Returns:
        Dictionary of metrics
    """
    points1 = get_trajectory_points(json1)
    points2 = get_trajectory_points(json2)

    seq1 = build_furniture_sequence(json1, yaml_data, merge=merge)
    seq2 = build_furniture_sequence(json2, yaml_data, merge=merge)

    edit_raw, edit_norm = compute_edit_distance(seq1, seq2)

    results = {
        'spatial_jsd': compute_spatial_jsd(points1, points2, grid_size=grid_size),
        'ngram_jsd_2': compute_ngram_jsd(seq1, seq2, n=2),
        'ngram_jsd_3': compute_ngram_jsd(seq1, seq2, n=3),
        'ngram_jsd_4': compute_ngram_jsd(seq1, seq2, n=4),
        'edit_dist_raw': edit_raw,
        'edit_dist_norm': edit_norm
    }

    return results


def run_single(
    json_path1: str,
    json_path2: str,
    yaml_path: str,
    merge: bool = False,
    grid_size: float = 3.0
) -> Dict[str, Union[int, float]]:
    """
    Run single comparison.

    Args:
        json_path1: First JSON file path
        json_path2: Second JSON file path
        yaml_path: YAML layout file path
        merge: Whether to merge consecutive same-type segments
        grid_size: Spatial grid size in decimeters

    Returns:
        Dictionary of metrics
    """
    json1 = load_json(json_path1)
    json2 = load_json(json_path2)
    yaml_data = load_yaml(yaml_path)

    return compute_all_metrics(json1, json2, yaml_data, merge=merge, grid_size=grid_size)


def main():
    parser = argparse.ArgumentParser(description='Compare diversity metrics between two trajectories')
    parser.add_argument('json1', help='First JSON trajectory file path')
    parser.add_argument('json2', help='Second JSON trajectory file path')
    parser.add_argument('-y', '--yaml', required=True, help='YAML layout file path')
    parser.add_argument('-m', '--merge', action='store_true',
                        help='Merge consecutive same-type segments')
    parser.add_argument('-g', '--grid-size', type=float, default=3.0,
                        help='Spatial grid size in decimeters, default 3.0 (30cm)')

    args = parser.parse_args()

    results = run_single(args.json1, args.json2, args.yaml,
                         merge=args.merge, grid_size=args.grid_size)

    print("=" * 50)
    print("Trajectory Diversity Metrics")
    print(f"  Merge consecutive: {'Yes' if args.merge else 'No'}")
    print(f"  Grid size: {args.grid_size} dm")
    print("=" * 50)
    print(f"Spatial JSD:           {results['spatial_jsd']:.6f}")
    print(f"2-gram JSD:            {results['ngram_jsd_2']:.6f}")
    print(f"3-gram JSD:            {results['ngram_jsd_3']:.6f}")
    print(f"4-gram JSD:            {results['ngram_jsd_4']:.6f}")
    print(f"Edit Distance (raw):   {results['edit_dist_raw']}")
    print(f"Edit Distance (norm):  {results['edit_dist_norm']:.6f}")
    print("=" * 50)

    return results


if __name__ == '__main__':
    main()
