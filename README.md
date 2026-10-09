# Solomon C1 Dynamic Layer — Baseline starter

This is a small, dependency-free Python baseline for testing a 5-stage workflow:

1. Load Solomon/Homberger XML instances.
2. Normalize source data and enrich tasks with dynamic attributes.
3. Simulate task releases/status and generate snapshots.
4. Assign tasks to capability-compatible UAVs using a simple nearest-task baseline.
5. Log snapshots, assignments, and summary metrics to JSON/CSV.

## Important modeling assumptions

- This is a baseline workflow test, not a competitive VRPTW/UAV optimizer.
- The XML parser recognizes the sample schema shown in the conversation:
  `<node id="0" type="0"><cx>40.0</cx><cy>50.0</cy></node>`,
  `<fleet><vehicle_profile ...>...</vehicle_profile></fleet>`,
  and `<request id="1" node="1"><tw><start>...</start><end>...</end>...</request>`.
- The parser is intentionally defensive, but XML dialects vary. If your files use different tags/attributes, adapt `xml_loader.py` after inspecting one complete file.
- The meaning of `type` values is not assumed.
- `tw/start` and `tw/end` are preserved as service time-window bounds. `release_time` is generated separately by the dynamic layer.
- Coordinate distances use Euclidean distance in the input coordinate units. They are not automatically meters or kilometers.
- The baseline assigns only capability-compatible tasks. It does not guarantee globally optimal routes or complete VRPTW feasibility.
- The simple baseline treats task service as starting only after both `release_time` and `ready_time`; it checks whether service can start by `due_time`. Change `deadline_mode` in config to `"completion"` if the task must finish by the deadline.

## Quick start

1. Put your 27 XML files into `data/instances/`.
2. Use Python 3.10+ (standard library only).
3. Edit `config.json`.
4. Run from this project directory:

   `python run_pipeline.py`

Outputs are written to `outputs/`:
- `normalized_instances.json`
- `snapshots.json`
- `assignments.csv`
- `run_summary.json`

## Expected XML shape

The loader searches the document for `node`, `vehicle_profile`, and `request` elements, including nested ones. A node should have `id`, `cx`, and `cy`; a request should have `id`, `node`, `tw/start`, `tw/end`, `quantity`, and `service_time`; a vehicle profile should have `number`, `departure_node`, `arrival_node`, `capacity`, and `max_travel_time`.

If the source XML uses different names or omits a field, inspect the error/warning and update `solomon_dynamic/xml_loader.py`.

## Configurable baseline rules

- `instance_glob`: file pattern under `instance_dir`
- `snapshot_interval`: simulation time step
- `simulation_end`: last simulation time
- `max_active_tasks`: maximum released but unfinished tasks in a snapshot
- `task_type_limits`: max active tasks per type
- `task_type_mix`: task-type probabilities assigned to requests
- `release_mode`: `"spread"` distributes releases across the horizon; `"zero"` releases all at time 0
- `release_seed`: random seed for repeatability
- `uav_profiles`: synthetic UAV capability profiles. These are research assumptions, not fields inferred from the original XML.
- `deadline_mode`: `"start"` checks service start <= due_time; `"completion"` checks service completion <= due_time

## Suggested first test

Start with one XML file and a short horizon. Check that:
- node and request counts look reasonable;
- every request references a known node;
- each snapshot has a valid timestamp;
- a task is never released before `release_time`;
- capability-incompatible UAVs never receive a task;
- expired tasks remain in the task history and are counted in metrics.

Then run all 27 instances.
