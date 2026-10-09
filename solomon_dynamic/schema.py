"""Validation and dynamic enrichment of a normalized source instance."""
from collections import Counter
import random


def validate_instance(instance):
    node_ids = {n["node_id"] for n in instance["nodes"]}
    request_ids = [r["request_id"] for r in instance["requests"]]
    errors = []
    if len(request_ids) != len(set(request_ids)):
        errors.append("Duplicate request IDs found.")
    for r in instance["requests"]:
        if r["node_id"] not in node_ids:
            errors.append(f"Request {r['request_id']} references unknown node {r['node_id']}.")
        if r["ready_time"] > r["due_time"]:
            errors.append(f"Request {r['request_id']} has ready_time > due_time.")
        if r["quantity"] < 0 or r["service_time"] < 0:
            errors.append(f"Request {r['request_id']} has negative quantity/service_time.")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "node_count": len(instance["nodes"]),
        "request_count": len(instance["requests"]),
        "vehicle_profile_count": len(instance["vehicle_profiles"]),
        "validation": "passed",
    }


def _choose_type(rng, type_mix):
    names = list(type_mix)
    weights = [float(type_mix[n]) for n in names]
    total = sum(weights)
    if not names or total <= 0:
        raise ValueError("task_type_mix must have at least one positive weight.")
    weights = [w / total for w in weights]
    return rng.choices(names, weights=weights, k=1)[0]


def enrich_instance(instance, config):
    """Add task_type, release_time, capability requirements and synthetic UAVs."""
    rng = random.Random(int(config.get("release_seed", 42)))
    start = float(config.get("simulation_start", 0))
    end = float(config.get("simulation_end", 1200))
    mode = config.get("release_mode", "spread")
    if mode not in {"spread", "zero"}:
        raise ValueError("release_mode must be 'spread' or 'zero'.")

    nodes = {n["node_id"]: n for n in instance["nodes"]}
    requests = sorted(instance["requests"], key=lambda r: int(r["request_id"]) if r["request_id"].isdigit() else r["request_id"])
    if mode == "spread" and len(requests) > 1:
        release_times = [
            start + i * (end - start) / (len(requests) - 1)
            for i in range(len(requests))
        ]
    elif mode == "spread":
        release_times = [start]
    else:
        release_times = [start] * len(requests)

    task_types = config["task_type_mix"]
    caps_by_type = config["default_task_capabilities"]
    service_by_type = config.get("task_service_time_by_type", {})
    tasks = []
    for i, (request, release_time) in enumerate(zip(requests, release_times)):
        task_type = _choose_type(rng, task_types)
        node = nodes[request["node_id"]]
        # Keep source window intact. Release time is an independent synthetic attribute.
        tasks.append({
            "task_id": f"{instance['instance_name']}_T{request['request_id']}",
            "source_request_id": request["request_id"],
            "node_id": request["node_id"],
            "x": node["x"],
            "y": node["y"],
            "task_type": task_type,
            "release_time": round(release_time, 3),
            "ready_time": request["ready_time"],
            "due_time": request["due_time"],
            "quantity": request["quantity"],
            "service_time": float(service_by_type.get(task_type, request["service_time"])),
            "required_capabilities": list(caps_by_type.get(task_type, [])),
            "status": "unreleased",
            "assigned_uav_id": None,
            "actual_start_time": None,
            "actual_completion_time": None,
            "outcome": None,
        })

    uavs = []
    uav_index = 1
    for profile in config["uav_profiles"]:
        for _ in range(int(profile["number"])):
            uavs.append({
                "uav_id": f"U{uav_index:03d}",
                "profile_id": profile["profile_id"],
                "capabilities": list(profile["capabilities"]),
                "x": nodes.get("0", {"x": 0, "y": 0})["x"],
                "y": nodes.get("0", {"x": 0, "y": 0})["y"],
                "speed": float(profile.get("speed", 1.0)),
                "capacity": float(profile.get("capacity", 0)),
                "max_travel_time": float(profile.get("max_travel_time", float("inf"))),
                "available_at": start,
                "current_task_id": None,
                "distance_travelled": 0.0,
                "busy_time": 0.0,
            })
            uav_index += 1

    return {"tasks": tasks, "uavs": uavs}
