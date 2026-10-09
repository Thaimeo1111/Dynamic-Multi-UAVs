"""Greedy capability-aware baseline planner.

This is deliberately simple: for each snapshot, assign pending tasks to the nearest
available compatible UAV, checking capacity and a basic travel-time/deadline rule.
It is not a full VRPTW solver and does not simulate flight execution between snapshots.
"""
from copy import deepcopy
import math
from .dynamic_layer import distance


def compatible(uav, task):
    return set(task["required_capabilities"]).issubset(set(uav["capabilities"]))


def plan_snapshot(snapshot, config):
    time_now = float(snapshot["time"])
    tasks = deepcopy(snapshot["tasks"])
    uavs = deepcopy(snapshot["uavs"])
    deadline_mode = config.get("deadline_mode", "start")
    records = []

    candidates = [
        task for task in tasks
        if task["status"] == "pending" and task["release_time"] <= time_now
    ]
    # Emergency first, then earliest due time, then stable task id.
    candidates.sort(key=lambda t: (
        0 if t["task_type"] == "emergency" else 1,
        t["due_time"],
        t["task_id"]
    ))

    for task in candidates:
        eligible = [
            u for u in uavs
            if compatible(u, task)
            and u["available_at"] <= time_now
            and task["quantity"] <= u["capacity"]
        ]
        if not eligible:
            records.append({
                "time": time_now, "task_id": task["task_id"],
                "task_type": task["task_type"], "assigned_uav_id": "",
                "status": "unassigned_no_eligible_uav", "distance": "",
                "estimated_arrival": "", "estimated_service_start": "",
                "estimated_completion": ""
            })
            continue

        # For baseline, UAVs begin each snapshot at their recorded coordinates.
        eligible.sort(key=lambda u: (distance(u, task), u["uav_id"]))
        uav = eligible[0]
        dist = distance(uav, task)
        speed = max(float(uav["speed"]), 1e-9)
        travel_time = dist / speed
        arrival = time_now + travel_time
        service_start = max(arrival, task["ready_time"])
        completion = service_start + task["service_time"]
        deadline_ok = service_start <= task["due_time"] if deadline_mode == "start" else completion <= task["due_time"]
        travel_ok = travel_time <= float(uav["max_travel_time"])
        if not deadline_ok or not travel_ok:
            status = "unassigned_infeasible_time"
            assigned = ""
        else:
            status = "assigned_baseline"
            assigned = uav["uav_id"]
            # This is a snapshot-level plan, not a persistent multi-stop route.
            # Reserve the UAV in this plan to avoid assigning multiple tasks at once.
            uav["available_at"] = completion
            uav["x"], uav["y"] = task["x"], task["y"]
            uav["distance_travelled"] += dist
            uav["busy_time"] += travel_time + task["service_time"]
        records.append({
            "time": time_now, "task_id": task["task_id"],
            "task_type": task["task_type"], "assigned_uav_id": assigned,
            "status": status, "distance": round(dist, 4),
            "estimated_arrival": round(arrival, 4),
            "estimated_service_start": round(service_start, 4),
            "estimated_completion": round(completion, 4)
        })
    return records
