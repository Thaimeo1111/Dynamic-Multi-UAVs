"""Defensive XML loader for the node/fleet/request schema shown in the project notes."""
from pathlib import Path
import xml.etree.ElementTree as ET


def _local(tag):
    """Remove an optional XML namespace prefix."""
    return tag.rsplit("}", 1)[-1].lower()


def _text(element, child_name, default=None):
    for child in element.iter():
        if _local(child.tag) == child_name.lower() and child is not element:
            if child.text and child.text.strip():
                return child.text.strip()
    return default


def _float(value, field, required=True, default=None):
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"Missing required numeric field: {field}")
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number for {field}: {value!r}") from exc


def load_xml_instance(path):
    """Load one XML instance into normalized dictionaries."""
    path = Path(path)
    root = ET.parse(path).getroot()

    nodes = {}
    for elem in root.iter():
        if _local(elem.tag) != "node":
            continue
        node_id = elem.attrib.get("id")
        if node_id is None:
            continue
        x = _text(elem, "cx")
        y = _text(elem, "cy")
        if x is None or y is None:
            # Avoid treating unrelated node-like tags as valid coordinate nodes.
            continue
        nodes[str(node_id)] = {
            "node_id": str(node_id),
            "node_type": elem.attrib.get("type"),
            "x": _float(x, f"node[{node_id}].cx"),
            "y": _float(y, f"node[{node_id}].cy"),
        }

    vehicle_profiles = []
    for elem in root.iter():
        if _local(elem.tag) != "vehicle_profile":
            continue
        attrs = elem.attrib
        vehicle_profiles.append({
            "source_type": attrs.get("type"),
            "number": int(_float(attrs.get("number"), "vehicle_profile.number")),
            "departure_node": str(_text(elem, "departure_node")),
            "arrival_node": str(_text(elem, "arrival_node")),
            "capacity": _float(_text(elem, "capacity"), "vehicle_profile.capacity"),
            "max_travel_time": _float(_text(elem, "max_travel_time"), "vehicle_profile.max_travel_time"),
        })

    requests = []
    for elem in root.iter():
        if _local(elem.tag) != "request":
            continue
        request_id = elem.attrib.get("id")
        node_id = elem.attrib.get("node")
        if request_id is None or node_id is None:
            continue
        tw_start = _text(elem, "start")
        tw_end = _text(elem, "end")
        quantity = _text(elem, "quantity")
        service_time = _text(elem, "service_time")
        if None in (tw_start, tw_end, quantity, service_time):
            raise ValueError(
                f"{path.name}: request id={request_id} is missing tw/start, tw/end, quantity, or service_time"
            )
        requests.append({
            "request_id": str(request_id),
            "node_id": str(node_id),
            "ready_time": _float(tw_start, f"request[{request_id}].tw.start"),
            "due_time": _float(tw_end, f"request[{request_id}].tw.end"),
            "quantity": _float(quantity, f"request[{request_id}].quantity"),
            "service_time": _float(service_time, f"request[{request_id}].service_time"),
        })

    if not nodes:
        raise ValueError(f"{path.name}: no nodes with id/cx/cy found")
    if not requests:
        raise ValueError(f"{path.name}: no valid request elements found")
    if not vehicle_profiles:
        raise ValueError(f"{path.name}: no vehicle_profile elements found")

    missing_nodes = sorted({r["node_id"] for r in requests if r["node_id"] not in nodes})
    if missing_nodes:
        raise ValueError(f"{path.name}: requests reference missing node IDs: {missing_nodes}")

    for profile in vehicle_profiles:
        for field in ("departure_node", "arrival_node"):
            if profile[field] not in nodes:
                raise ValueError(
                    f"{path.name}: vehicle profile {field}={profile[field]} is not in parsed nodes"
                )

    return {
        "instance_name": path.stem,
        "source_path": str(path),
        "nodes": list(nodes.values()),
        "vehicle_profiles": vehicle_profiles,
        "requests": requests,
        "warnings": [
            "Source node/vehicle type codes are preserved but not interpreted.",
            "Request time windows are preserved; dynamic release times are added separately."
        ],
    }
