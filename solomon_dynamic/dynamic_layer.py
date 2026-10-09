"""Task lifecycle and snapshot generation."""
from copy import deepcopy
import math


def distance(a, b):
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def generate_snapshots(instance, enriched, config):
    """Generate simple snapshots. A task is released at release_time and expires after its due time."""
    tasks = deepcopy(enriched["tasks"])
    uavs = deepcopy(enriched["uavs"])
    node_map = {n["node_id"]: n for n in instance["nodes"]}
    interval = float(config["snapshot_interval"])
    start = float(config.get("simulation_start", 0))
    end = float(config.get("simulation_end", 1200))
    max_active = int(config.get("max_active_tasks", 10))
    type_limits = config.get("task_type_limits", {})
    if interval <= 0 or end < start:
        raise ValueError("Invalid simulation interval or horizon.")

    snapshots = []
    t = start
    while t <= end + 1e-9:
        # Release due tasks, subject to active-task and per-type caps.
        active = [task for task in tasks if task["status"] in {"pending", "assigned", "in_progress"}]
        for task in tasks:
            if task["status"] != "unreleased" or task["release_time"] > t:
                continue
            active = [x for x in tasks if x["status"] in {"pending", "assigned", "in_progress"}]
            same_type_count = sum(1 for x in active if x["task_type"] == task["task_type"])
            if len(active) >= max_active:
                task["status"] = "deferred_by_capacity"
                task["outcome"] = "not_released_due_to_active_cap"
                continue
            if task["task_type"] in type_limits and same_type_count >= int(type_limits[task["task_type"]]):
                task["status"] = "deferred_by_type_cap"
                task["outcome"] = "not_released_due_to_type_cap"
                continue
            task["status"] = "pending"

        # Expire only tasks that have been released and remain unfinished.
        deadline_mode = config.get("deadline_mode", "start")
        for task in tasks:
            if task["status"] not in {"pending", "assigned"}:
                continue
            deadline = task["due_time"]
            if deadline_mode == "start":
                expired = t > deadline
            else:
                expired = t > deadline
            if expired:
                task["status"] = "expired"
                task["outcome"] = "expired_unserved"

        active_tasks = [
            task for task in tasks
            if task["status"] in {"pending", "assigned", "in_progress"}
        ]
        snapshots.append({
            "time": round(t, 3),
            "instance_name": instance["instance_name"],
            "active_task_ids": [task["task_id"] for task in active_tasks],
            "active_task_count": len(active_tasks),
            "active_task_counts_by_type": {
                kind: sum(1 for task in active_tasks if task["task_type"] == kind)
                for kind in sorted({task["task_type"] for task in tasks})
            },
            "tasks": deepcopy(tasks),
            "uavs": deepcopy(uavs),
        })
        t += interval

    return {"snapshots": snapshots, "final_tasks": tasks, "uavs": uavs}
