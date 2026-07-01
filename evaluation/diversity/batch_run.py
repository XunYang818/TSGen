"""Batch run: compute diversity metrics for all layouts in directory."""
import argparse
import os
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from typing import Dict, List, Tuple
import statistics

from .tools import load_json, load_yaml
from .single_run import compute_all_metrics

GRID_SIZE = 3.0


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Parse filename to extract layout name and occupation.
    Format: yaml_filename-occupation.json (split by first -)

    Args:
        filename: Filename (e.g., "1-Teacher.json")

    Returns:
        (layout_name, occupation) tuple
    """
    name = filename.rsplit('.', 1)[0]
    parts = name.split('-', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, ''


def group_files_by_layout(input_dir: str) -> Dict[str, List[str]]:
    """
    Group JSON files by layout.

    Args:
        input_dir: Input directory path

    Returns:
        Dictionary mapping layout_name -> list of file paths
    """
    groups = defaultdict(list)
    for f in os.listdir(input_dir):
        if f.endswith('.json'):
            layout, _ = parse_filename(f)
            groups[layout].append(os.path.join(input_dir, f))
    return dict(groups)


def compute_pairwise_metrics_both(
    files: List[str],
    yaml_path: str,
    show_progress: bool = True,
    grid_size: float = 3.0
) -> List[Dict]:
    """
    Compute metrics for all pairs (both merge=False and merge=True, combined in one row).

    Args:
        files: List of JSON file paths
        yaml_path: YAML layout file path
        show_progress: Whether to show progress
        grid_size: Spatial grid size in decimeters

    Returns:
        List of combined results
    """
    yaml_data = load_yaml(yaml_path)
    json_cache = {f: load_json(f) for f in files}

    pairs = list(combinations(files, 2))
    total = len(pairs)
    results = []

    for idx, (f1, f2) in enumerate(pairs):
        if show_progress and (idx + 1) % 50 == 0:
            print(f"  Progress: {idx + 1}/{total}", flush=True)

        m_no = compute_all_metrics(json_cache[f1], json_cache[f2], yaml_data,
                                   merge=False, grid_size=grid_size)
        m_yes = compute_all_metrics(json_cache[f1], json_cache[f2], yaml_data,
                                    merge=True, grid_size=grid_size)

        results.append({
            'file1': os.path.basename(f1),
            'file2': os.path.basename(f2),
            'no_spatial_jsd': m_no['spatial_jsd'],
            'no_ngram_jsd_2': m_no['ngram_jsd_2'],
            'no_ngram_jsd_3': m_no['ngram_jsd_3'],
            'no_ngram_jsd_4': m_no['ngram_jsd_4'],
            'no_edit_dist_raw': m_no['edit_dist_raw'],
            'no_edit_dist_norm': m_no['edit_dist_norm'],
            'mg_spatial_jsd': m_yes['spatial_jsd'],
            'mg_ngram_jsd_2': m_yes['ngram_jsd_2'],
            'mg_ngram_jsd_3': m_yes['ngram_jsd_3'],
            'mg_ngram_jsd_4': m_yes['ngram_jsd_4'],
            'mg_edit_dist_raw': m_yes['edit_dist_raw'],
            'mg_edit_dist_norm': m_yes['edit_dist_norm'],
        })

    return results


def compute_average_both(results: List[Dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute average metrics for both modes."""
    if not results:
        return {}, {}

    avg_no = {}
    for key in ['spatial_jsd', 'ngram_jsd_2', 'ngram_jsd_3', 'ngram_jsd_4', 'edit_dist_raw', 'edit_dist_norm']:
        values = [r[f'no_{key}'] for r in results]
        avg_no[key] = statistics.mean(values) if values else 0.0

    avg_mg = {}
    for key in ['spatial_jsd', 'ngram_jsd_2', 'ngram_jsd_3', 'ngram_jsd_4', 'edit_dist_raw', 'edit_dist_norm']:
        values = [r[f'mg_{key}'] for r in results]
        avg_mg[key] = statistics.mean(values) if values else 0.0

    return avg_no, avg_mg


def format_results_combined(
    layout: str,
    pairwise: List[Dict],
    avg_no: Dict[str, float],
    avg_mg: Dict[str, float]
) -> str:
    """Format results for single layout (combined table)."""
    lines = []
    lines.append(f"Layout: {layout}")
    lines.append("=" * 180)

    lines.append("\nPairwise Comparison (no_=no merge, mg_=merged):")
    lines.append("-" * 180)
    header = (f"{'File1':<25} {'File2':<25} "
              f"{'no_Spt':>7} {'no_2g':>7} {'no_3g':>7} {'no_4g':>7} {'no_EdR':>7} {'no_EdN':>7} "
              f"{'mg_Spt':>7} {'mg_2g':>7} {'mg_3g':>7} {'mg_4g':>7} {'mg_EdR':>7} {'mg_EdN':>7}")
    lines.append(header)
    lines.append("-" * 180)

    for r in pairwise:
        line = (f"{r['file1']:<25} {r['file2']:<25} "
                f"{r['no_spatial_jsd']:>7.4f} {r['no_ngram_jsd_2']:>7.4f} {r['no_ngram_jsd_3']:>7.4f} "
                f"{r['no_ngram_jsd_4']:>7.4f} {r['no_edit_dist_raw']:>7} {r['no_edit_dist_norm']:>7.4f} "
                f"{r['mg_spatial_jsd']:>7.4f} {r['mg_ngram_jsd_2']:>7.4f} {r['mg_ngram_jsd_3']:>7.4f} "
                f"{r['mg_ngram_jsd_4']:>7.4f} {r['mg_edit_dist_raw']:>7} {r['mg_edit_dist_norm']:>7.4f}")
        lines.append(line)

    lines.append("-" * 180)
    lines.append(f"\nLayout Average:")
    lines.append(f"  {'Metric':<20} {'No Merge':>12} {'Merged':>12}")
    lines.append(f"  {'-'*44}")
    lines.append(f"  {'Spatial JSD':<20} {avg_no.get('spatial_jsd', 0):>12.6f} {avg_mg.get('spatial_jsd', 0):>12.6f}")
    lines.append(f"  {'2-gram JSD':<20} {avg_no.get('ngram_jsd_2', 0):>12.6f} {avg_mg.get('ngram_jsd_2', 0):>12.6f}")
    lines.append(f"  {'3-gram JSD':<20} {avg_no.get('ngram_jsd_3', 0):>12.6f} {avg_mg.get('ngram_jsd_3', 0):>12.6f}")
    lines.append(f"  {'4-gram JSD':<20} {avg_no.get('ngram_jsd_4', 0):>12.6f} {avg_mg.get('ngram_jsd_4', 0):>12.6f}")
    lines.append(f"  {'Edit Distance (Raw)':<20} {avg_no.get('edit_dist_raw', 0):>12.2f} {avg_mg.get('edit_dist_raw', 0):>12.2f}")
    lines.append(f"  {'Edit Distance (Norm)':<20} {avg_no.get('edit_dist_norm', 0):>12.6f} {avg_mg.get('edit_dist_norm', 0):>12.6f}")
    lines.append("")

    return '\n'.join(lines)


def format_total_average_combined(total_avg_no: Dict[str, float], total_avg_mg: Dict[str, float]) -> str:
    """Format total average (combined modes)."""
    lines = []
    lines.append("=" * 180)
    lines.append("Total Average Across All Layouts")
    lines.append("=" * 180)
    lines.append(f"  {'Metric':<20} {'No Merge':>12} {'Merged':>12}")
    lines.append(f"  {'-'*44}")
    lines.append(f"  {'Spatial JSD':<20} {total_avg_no.get('spatial_jsd', 0):>12.6f} {total_avg_mg.get('spatial_jsd', 0):>12.6f}")
    lines.append(f"  {'2-gram JSD':<20} {total_avg_no.get('ngram_jsd_2', 0):>12.6f} {total_avg_mg.get('ngram_jsd_2', 0):>12.6f}")
    lines.append(f"  {'3-gram JSD':<20} {total_avg_no.get('ngram_jsd_3', 0):>12.6f} {total_avg_mg.get('ngram_jsd_3', 0):>12.6f}")
    lines.append(f"  {'4-gram JSD':<20} {total_avg_no.get('ngram_jsd_4', 0):>12.6f} {total_avg_mg.get('ngram_jsd_4', 0):>12.6f}")
    lines.append(f"  {'Edit Distance (Raw)':<20} {total_avg_no.get('edit_dist_raw', 0):>12.2f} {total_avg_mg.get('edit_dist_raw', 0):>12.2f}")
    lines.append(f"  {'Edit Distance (Norm)':<20} {total_avg_no.get('edit_dist_norm', 0):>12.6f} {total_avg_mg.get('edit_dist_norm', 0):>12.6f}")
    return '\n'.join(lines)


def run_batch(input_dir: str, yaml_dir: str, grid_size: float = 3.0) -> str:
    """
    Batch run diversity evaluation.

    Args:
        input_dir: Input JSON files directory
        yaml_dir: YAML layout files directory
        grid_size: Spatial grid size in decimeters

    Returns:
        Result text
    """
    groups = group_files_by_layout(input_dir)

    layout_results = []
    output_lines = []

    output_lines.append(f"Config: grid_size={grid_size}dm")
    output_lines.append("Output both no-merge (no_) and merged (mg_) results")
    output_lines.append("")

    for layout, files in sorted(groups.items()):
        yaml_path = os.path.join(yaml_dir, f"{layout}.yaml")
        if not os.path.exists(yaml_path):
            print(f"Warning: Layout file {yaml_path} not found, skipping layout {layout}")
            continue

        if len(files) < 2:
            print(f"Warning: Layout {layout} has less than 2 files, skipping")
            continue

        print(f"Processing layout: {layout} ({len(files)} files)", flush=True)

        pairwise = compute_pairwise_metrics_both(files, yaml_path, grid_size=grid_size)
        avg_no, avg_mg = compute_average_both(pairwise)

        layout_results.append((layout, pairwise, avg_no, avg_mg))

    for layout, pairwise, avg_no, avg_mg in layout_results:
        output_lines.append(format_results_combined(layout, pairwise, avg_no, avg_mg))

    if layout_results:
        total_avg_no = {}
        total_avg_mg = {}
        for key in ['spatial_jsd', 'ngram_jsd_2', 'ngram_jsd_3', 'ngram_jsd_4', 'edit_dist_raw', 'edit_dist_norm']:
            values_no = [avg_no[key] for _, _, avg_no, _ in layout_results if key in avg_no]
            values_mg = [avg_mg[key] for _, _, _, avg_mg in layout_results if key in avg_mg]
            total_avg_no[key] = statistics.mean(values_no) if values_no else 0.0
            total_avg_mg[key] = statistics.mean(values_mg) if values_mg else 0.0
        output_lines.append(format_total_average_combined(total_avg_no, total_avg_mg))

    return '\n'.join(output_lines)


def main():
    parser = argparse.ArgumentParser(description='Batch compute trajectory diversity metrics')
    parser.add_argument('-i', '--input', required=True, help='Input JSON files directory')
    parser.add_argument('-y', '--yaml', required=True, help='YAML layout files directory')
    parser.add_argument('-g', '--grid-size', type=float, default=3.0,
                        help='Spatial grid size in decimeters, default 3.0 (30cm)')

    args = parser.parse_args()

    results = run_batch(args.input, args.yaml, grid_size=args.grid_size)

    input_dir_name = os.path.basename(os.path.normpath(args.input))
    grid_flag = f"g{args.grid_size:.0f}" if args.grid_size == int(args.grid_size) else f"g{args.grid_size}"
    output_file = f"{input_dir_name}_{grid_flag}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(results)

    print(f"\nResults saved to: {output_file}")
    print("\n" + results)


if __name__ == '__main__':
    main()
