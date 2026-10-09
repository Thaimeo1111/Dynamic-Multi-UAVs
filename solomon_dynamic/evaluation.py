"""Output writers and basic metrics."""
import csv
import json
from pathlib import Path
from collections import Counter


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_instance(instance_name, tasks, assignments, snapshots):
    released = [t for t in tasks if t["status"] != "unreleased"]
    completed = [t for t in tasks if t.get("status") == "completed"]
    expired = [t for t in tasks if t.get("status") == "expired"]
    assigned_rows = [r for r in assignments if r["status"] == "assigned_baseline"]
    unassigned = [r for r in assignments if r["status"].startswith("unassigned")]
    counts = Counter(t["task_type"] for t in tasks)
    return {
        "instance_name": instance_name,
        "node_count": None,
        "source_request_count": len(tasks),
        "task_type_counts_total": dict(counts),
        "released_or_terminal_count": len(released),
        "completed_count": len(completed),
        "expired_count": len(expired),
        "baseline_assignment_rows": len(assigned_rows),
        "unassigned_rows": len(unassigned),
        "snapshot_count": len(snapshots),
        "note": "Baseline planning assignments are estimates; this version does not yet execute persistent routes across time."
    }
