"""Analyze LLM call statistics from JSON files in directory."""
import json
import sys
from pathlib import Path


def load_json_data(filepath: str) -> dict | None:
    """Load complete JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Cannot read {filepath}: {e}")
        return None


def count_merged_actions(action_steps: list) -> int:
    """Count merged actions (consecutive same-type treated as one)."""
    if not action_steps:
        return 0

    merged_count = 1
    prev_type = action_steps[0].get('action_type', '')

    for action in action_steps[1:]:
        curr_type = action.get('action_type', '')
        if curr_type != prev_type:
            merged_count += 1
            prev_type = curr_type

    return merged_count


def extract_stats(data: dict) -> dict:
    """Extract raw statistics from JSON data."""
    info = data.get('info', {})
    usage = info.get('usage', {})
    agents = usage.get('agents', {})
    pass_once = usage.get('pass_once', {})
    counts = info.get('counts', {})

    action_steps = data.get('action_steps', [])
    merged_actions = count_merged_actions(action_steps)

    def get_agent(name):
        a = agents.get(name, {})
        return {
            'calls': a.get('calls', 0),
            'input_tokens': a.get('input_tokens', 0),
            'output_tokens': a.get('output_tokens', 0),
            'total_tokens': a.get('total_tokens', 0)
        }

    plan_gen = get_agent('plan_gen')
    plan_val = get_agent('plan_val')
    plan_cor = get_agent('plan_cor')

    event_gen = get_agent('event_gen')
    event_val = get_agent('event_val')
    event_cor = get_agent('event_cor')

    action_gen = get_agent('action_gen')
    action_val = get_agent('action_val')
    action_cor = get_agent('action_cor')

    return {
        'plan': {
            'gen_calls': plan_gen['calls'],
            'cor_calls': plan_cor['calls'],
            'pass_once': pass_once.get('plan', 0),
            'input_tokens': plan_gen['input_tokens'] + plan_val['input_tokens'] + plan_cor['input_tokens'],
            'output_tokens': plan_gen['output_tokens'] + plan_val['output_tokens'] + plan_cor['output_tokens'],
            'count': counts.get('plans', 0)
        },
        'event': {
            'gen_calls': event_gen['calls'],
            'cor_calls': event_cor['calls'],
            'pass_once': pass_once.get('event', 0),
            'input_tokens': event_gen['input_tokens'] + event_val['input_tokens'] + event_cor['input_tokens'],
            'output_tokens': event_gen['output_tokens'] + event_val['output_tokens'] + event_cor['output_tokens'],
            'count': counts.get('events', 0)
        },
        'action': {
            'gen_calls': action_gen['calls'],
            'cor_calls': action_cor['calls'],
            'pass_once': pass_once.get('action', 0),
            'input_tokens': action_gen['input_tokens'] + action_val['input_tokens'] + action_cor['input_tokens'],
            'output_tokens': action_gen['output_tokens'] + action_val['output_tokens'] + action_cor['output_tokens'],
            'count': counts.get('actions', 0),
            'merged_count': merged_actions
        }
    }


def aggregate_stats(all_stats: list[dict]) -> dict:
    """Aggregate statistics from multiple files."""
    n = len(all_stats)
    if n == 0:
        return {}

    categories = ['plan', 'event', 'action']
    fields = ['gen_calls', 'cor_calls', 'pass_once', 'input_tokens', 'output_tokens', 'count']

    totals = {cat: {f: 0 for f in fields} for cat in categories}
    for s in all_stats:
        for cat in categories:
            for f in fields:
                totals[cat][f] += s[cat][f]

    total_merged = sum(s['action'].get('merged_count', 0) for s in all_stats)

    totals['total'] = {f: sum(totals[cat][f] for cat in categories) for f in fields}

    result = {}
    for cat in categories + ['total']:
        raw = totals[cat]
        gen = raw['gen_calls']
        cor = raw['cor_calls']
        po = raw['pass_once']
        error_count = gen - po

        result[cat] = {
            'input_tokens': raw['input_tokens'] / n,
            'output_tokens': raw['output_tokens'] / n,
            'total_tokens': (raw['input_tokens'] + raw['output_tokens']) / n,
            'success_rate': gen / (gen + cor) if (gen + cor) > 0 else 1.0,
            'pass_once_rate': po / gen if gen > 0 else 1.0,
            'avg_corrections': cor / error_count if error_count > 0 else 0.0,
            'count': raw['count'] / n
        }

    result['action']['merged_count'] = total_merged / n

    return result


def format_output(agg: dict, file_count: int) -> str:
    """Format output."""
    lines = [
        f"Files analyzed: {file_count}",
        "=" * 50,
        ""
    ]

    categories = [('plan', 'Plan'), ('event', 'Event'), ('action', 'Action'), ('total', 'Total')]

    for key, name in categories:
        m = agg[key]
        lines.append(f"[{name}]")
        lines.append(f"  Input Tokens:       {m['input_tokens']:,.0f}")
        lines.append(f"  Output Tokens:      {m['output_tokens']:,.0f}")
        lines.append(f"  Total Tokens:       {m['total_tokens']:,.0f}")
        lines.append(f"  Success Rate:       {m['success_rate']*100:.2f}%")
        lines.append(f"  Pass-once Rate:     {m['pass_once_rate']*100:.2f}%")
        lines.append(f"  Avg Corrections:    {m['avg_corrections']:.2f}")
        lines.append(f"  Count:              {m['count']:.1f}")
        if key == 'action' and 'merged_count' in m:
            lines.append(f"  Merged Count:       {m['merged_count']:.1f}")
        lines.append("")

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_stats.py <directory_path>")
        sys.exit(1)

    target_dir = Path(sys.argv[1])
    if not target_dir.is_dir():
        print(f"Error: {target_dir} is not a valid directory")
        sys.exit(1)

    json_files = list(target_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files in directory {target_dir}")
        sys.exit(1)

    all_stats = []
    for fp in json_files:
        data = load_json_data(fp)
        if data:
            stats = extract_stats(data)
            all_stats.append(stats)
            print(f"Processed: {fp.name}")

    if not all_stats:
        print("No valid statistics data")
        sys.exit(1)

    agg = aggregate_stats(all_stats)
    output = format_output(agg, len(all_stats))

    output_file = Path(f"{target_dir.name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n{output}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
