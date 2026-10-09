import argparse
import json
from pathlib import Path

from solomon_dynamic.xml_loader import load_xml_instance
from solomon_dynamic.schema import validate_instance, enrich_instance
from solomon_dynamic.dynamic_layer import generate_snapshots
from solomon_dynamic.baseline_planner import plan_snapshot
from solomon_dynamic.evaluation import write_json, write_csv, summarize_instance


def main():
    parser = argparse.ArgumentParser(description="Run Solomon XML dynamic-layer baseline.")
    parser.add_argument("--config", default="config.json", help="Path to JSON config.")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    input_dir = Path(config["instance_dir"])
    files = sorted(input_dir.glob(config.get("instance_glob", "*.xml")))
    if not files:
        raise SystemExit(
            f"No XML files found in {input_dir.resolve()}. Put your 27 XML files there or change config.json."
        )

    output_dir = Path(config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_normalized = []
    all_snapshots = []
    all_assignments = []
    summaries = []
    errors = []

    for xml_path in files:
        print(f"[LOAD] {xml_path.name}")
        try:
            instance = load_xml_instance(xml_path)
            validation = validate_instance(instance)
            enriched = enrich_instance(instance, config)
            dynamic = generate_snapshots(instance, enriched, config)

            instance_assignments = []
            for snapshot in dynamic["snapshots"]:
                instance_assignments.extend(plan_snapshot(snapshot, config))

            summary = summarize_instance(
                instance["instance_name"], dynamic["final_tasks"],
                instance_assignments, dynamic["snapshots"]
            )
            summary["node_count"] = validation["node_count"]
            summary["validation"] = validation["validation"]
            summaries.append(summary)
            all_normalized.append({
                "instance_name": instance["instance_name"],
                "source_path": instance["source_path"],
                "validation": validation,
                "nodes": instance["nodes"],
                "vehicle_profiles": instance["vehicle_profiles"],
                "tasks": enriched["tasks"],
                "uavs": enriched["uavs"],
            })
            all_snapshots.append({
                "instance_name": instance["instance_name"],
                "snapshots": dynamic["snapshots"]
            })
            all_assignments.extend([
                {"instance_name": instance["instance_name"], **row}
                for row in instance_assignments
            ])
            print(
                f"  nodes={validation['node_count']}, requests={validation['request_count']}, "
                f"snapshots={len(dynamic['snapshots'])}"
            )
        except Exception as exc:
            errors.append({"file": xml_path.name, "error": str(exc)})
            print(f"[ERROR] {xml_path.name}: {exc}")

    write_json(output_dir / "normalized_instances.json", all_normalized)
    write_json(output_dir / "snapshots.json", all_snapshots)
    write_csv(output_dir / "assignments.csv", all_assignments)
    write_json(output_dir / "run_summary.json", {
        "input_file_count": len(files),
        "successful_instances": len(summaries),
        "failed_instances": len(errors),
        "summaries": summaries,
        "errors": errors
    })

    print("\n=== FINISHED ===")
    print(f"Input files: {len(files)}")
    print(f"Successful: {len(summaries)}")
    print(f"Failed: {len(errors)}")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
