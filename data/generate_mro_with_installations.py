#!/usr/bin/env python3
"""Generate synthetic aircraft MRO snapshots with two removal lifecycle cases.

A: Unscheduled removal -> different replacement component installed.
B: Unscheduled removal -> temporary loaner -> same physical component repaired
   and reinstalled under a NEW installation_id.

The target is an unscheduled removal of the CURRENT INSTALLATION within the
NEXT 30 AIRCRAFT FLIGHT CYCLES, not the next 30 calendar days.

IMPORTANT: *_AUDIT_ONLY.csv files contain future events. Never join them into
model features. They exist solely for dataset verification and label creation.
The `label_available_at` column is split metadata only, never a model feature.
The simulator assumes no additional removal events through the end of each
record's 30-cycle follow-up, even if no later snapshot is exported.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

N_AIRCRAFT = 100
POSITIONS_PER_AIRCRAFT = 5
SNAPSHOTS_PER_POSITION = 20
CYCLES_PER_SNAPSHOT = 15
HORIZON_CYCLES = 30
SNAPSHOT_DAYS = 7
SPLIT_AIRCRAFT = {"train": 70, "validation": 15, "test": 15}
SPLIT_EVENTS = {"train": 70, "validation": 15, "test": 15}
SPLIT_BASE_DATE = {
    "train": datetime(2022, 1, 3),
    "validation": datetime(2023, 1, 2),
    "test": datetime(2024, 1, 1),
}

COLUMNS = [
    "record_id", "snapshot_date", "aircraft_id", "aircraft_model",
    "component_id", "installation_id", "component_type", "component_position",
    "flight_cycles", "cycles_since_install", "vibration_rms", "temperature_c",
    "pressure_delta", "fault_count_30d", "maintenance_count_90d",
    "last_maintenance_cycles_ago", "fault_code",
    "unscheduled_removal_next_30_cycles", "label_available_at", "split",
]
EVENT_COLUMNS = [
    "aircraft_id", "component_id", "installation_id", "component_type",
    "component_position", "removal_cycle", "removal_date", "removal_type",
    "lifecycle_case", "split",
]
INSTALL_COLUMNS = [
    "aircraft_id", "component_id", "installation_id", "component_position",
    "installation_cycle", "installation_date", "installation_reason", "split",
]


def weighted_sample_distinct(rng: random.Random, objects: list[dict], n: int) -> list[dict]:
    if n > len(objects):
        raise ValueError("Cannot sample more unique slots than available")
    keys = [(rng.random() ** (1.0 / obj["weight"]), obj) for obj in objects]
    return [obj for _, obj in sorted(keys, key=lambda pair: pair[0], reverse=True)[:n]]


def poisson(rng: random.Random, lam: float) -> int:
    threshold = math.exp(-lam)
    k, product = 0, 1.0
    while product > threshold:
        k += 1
        product *= rng.random()
    return k - 1


def wear_at(slot: dict, t: int) -> float:
    """Unobserved wear state (not an exported model feature)."""
    return max(0.02, min(1.35, slot["base_wear"] + t * slot["wear_slope"]
                               + 0.015 * math.sin(t / 3.0 + slot["phase"])))


def blank_with_probability(rng: random.Random, value: object, p: float) -> object:
    return "" if rng.random() < p else value


def date_for_cycle(slot: dict, cycle: int) -> datetime:
    """Simulator maps each 15 cycles to 7 days for calendar timestamps."""
    return slot["start_date"] + timedelta(
        days=SNAPSHOT_DAYS * (cycle - slot["start_cycles"]) / CYCLES_PER_SNAPSHOT
    )


def write_csv(path: Path, fields: list[str], records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_dataset(seed: int = 42, out_dir: Path = Path("data")) -> tuple[Path, Path, Path]:
    rng = random.Random(seed)
    slots: list[dict] = []
    next_component_id = 0
    next_installation_id = 0

    def new_component_id() -> str:
        nonlocal next_component_id
        next_component_id += 1
        return f"COMP-{next_component_id:06d}"

    def new_installation_id() -> str:
        nonlocal next_installation_id
        next_installation_id += 1
        return f"INST-{next_installation_id:06d}"

    # Create 100 aircraft x 5 positions, one physical component per slot.
    aircraft_idx = 0
    for split, n_aircraft in SPLIT_AIRCRAFT.items():
        for _ in range(n_aircraft):
            aircraft_idx += 1
            aircraft_id = f"AC-{aircraft_idx:04d}"
            model = rng.choice(["A320", "B737", "B787", "A350"])
            start_date = SPLIT_BASE_DATE[split] + timedelta(days=rng.randint(0, 14))
            start_cycles = rng.randint(4500, 24000)
            for pos, component_type in enumerate(
                ["pump", "valve", "actuator", "sensor", "compressor"], start=1
            ):
                base_wear = rng.uniform(0.07, 0.55)
                slope = rng.uniform(0.006, 0.031)
                slot = {
                    "aircraft_id": aircraft_id,
                    "aircraft_model": model,
                    "component_type": component_type,
                    "position": pos,
                    "start_date": start_date,
                    "start_cycles": start_cycles,
                    "base_age": rng.randint(150, min(5200, start_cycles - 1)),
                    "base_wear": base_wear,
                    "wear_slope": slope,
                    "phase": rng.uniform(0, math.tau),
                    "split": split,
                    "weight": 0.6 + 7.0 * (base_wear + 12 * slope) ** 2,
                    "original_component_id": new_component_id(),
                    "original_installation_id": new_installation_id(),
                }
                slots.append(slot)

    # Choose 100 UNSCHEDULED removals, each giving exactly two positive
    # weekly snapshots at cycle spacing 15 and target horizon 30.
    for split, n_events in SPLIT_EVENTS.items():
        subset = [s for s in slots if s["split"] == split]
        selected = weighted_sample_distinct(rng, subset, n_events)
        rng.shuffle(selected)
        # Roughly half of removals lead to the original part returning later.
        for index, slot in enumerate(selected):
            case = "repair_reinstall" if index < n_events // 2 else "replacement"
            slot["lifecycle_case"] = case
            # Reinstall cases must leave room for the repair / second installation.
            candidates = range(4, 15) if case == "repair_reinstall" else range(4, 19)
            weights = [0.15 + wear_at(slot, t) ** 2.0 for t in candidates]
            after_t = rng.choices(list(candidates), weights=weights, k=1)[0]
            offset = rng.randint(1, CYCLES_PER_SNAPSHOT - 1)
            event_cycle = slot["start_cycles"] + after_t * CYCLES_PER_SNAPSHOT + offset
            slot["event_cycle"] = event_cycle
            slot["secondary_component_id"] = new_component_id()
            slot["secondary_installation_id"] = new_installation_id()
            if case == "repair_reinstall":
                return_t = after_t + rng.choice([2, 3])
                # Return event occurs between the final loaner snapshot and the
                # first repaired-component snapshot.
                return_cycle = (
                    slot["start_cycles"]
                    + return_t * CYCLES_PER_SNAPSHOT
                    - rng.randint(1, CYCLES_PER_SNAPSHOT - 1)
                )
                slot["return_cycle"] = return_cycle
                slot["return_installation_id"] = new_installation_id()

    removal_events: list[dict] = []
    installation_events: list[dict] = []

    # Keep installation timelines in separate AUDIT_ONLY exports.
    for slot in slots:
        common = {
            "aircraft_id": slot["aircraft_id"],
            "component_position": slot["position"],
            "split": slot["split"],
        }
        initial_install_cycle = slot["start_cycles"] - slot["base_age"]
        installation_events.append({
            **common,
            "component_id": slot["original_component_id"],
            "installation_id": slot["original_installation_id"],
            "installation_cycle": initial_install_cycle,
            "installation_date": "",  # before observed history: date unknown
            "installation_reason": "INITIAL_PREEXISTING",
        })
        if "event_cycle" not in slot:
            continue
        scenario = slot["lifecycle_case"]
        removal_events.append({
            **common,
            "component_id": slot["original_component_id"],
            "installation_id": slot["original_installation_id"],
            "component_type": slot["component_type"],
            "removal_cycle": slot["event_cycle"],
            "removal_date": date_for_cycle(slot, slot["event_cycle"]).date().isoformat(),
            "removal_type": "UNSCHEDULED",
            "lifecycle_case": scenario,
        })
        installation_events.append({
            **common,
            "component_id": slot["secondary_component_id"],
            "installation_id": slot["secondary_installation_id"],
            "installation_cycle": slot["event_cycle"],
            "installation_date": date_for_cycle(slot, slot["event_cycle"]).date().isoformat(),
            "installation_reason": "NEW_REPLACEMENT" if scenario == "replacement" else "LOANER",
        })
        if scenario == "repair_reinstall":
            removal_events.append({
                **common,
                "component_id": slot["secondary_component_id"],
                "installation_id": slot["secondary_installation_id"],
                "component_type": slot["component_type"],
                "removal_cycle": slot["return_cycle"],
                "removal_date": date_for_cycle(slot, slot["return_cycle"]).date().isoformat(),
                "removal_type": "SCHEDULED",
                "lifecycle_case": "loaner_return",
            })
            installation_events.append({
                **common,
                "component_id": slot["original_component_id"],
                "installation_id": slot["return_installation_id"],
                "installation_cycle": slot["return_cycle"],
                "installation_date": date_for_cycle(slot, slot["return_cycle"]).date().isoformat(),
                "installation_reason": "REINSTALL_AFTER_REPAIR",
            })

    # Labels must be keyed to INSTALLATIONS, not just physical component IDs.
    removal_by_install = {e["installation_id"]: e for e in removal_events}
    unscheduled_by_install = {
        e["installation_id"]: int(e["removal_cycle"])
        for e in removal_events if e["removal_type"] == "UNSCHEDULED"
    }
    records: list[dict] = []
    for slot in slots:
        fault_history: list[tuple[datetime, int]] = []
        maintenance_history: list[tuple[datetime, int]] = []
        current_installation = None
        last_maintenance_cycle: int | None = None

        for t in range(SNAPSHOTS_PER_POSITION):
            current_cycle = slot["start_cycles"] + t * CYCLES_PER_SNAPSHOT
            date = slot["start_date"] + timedelta(days=t * SNAPSHOT_DAYS)
            event_cycle = slot.get("event_cycle")

            if event_cycle is None or current_cycle < event_cycle:
                component_id = slot["original_component_id"]
                installation_id = slot["original_installation_id"]
                install_cycle = slot["start_cycles"] - slot["base_age"]
                wear = wear_at(slot, t)
            elif (
                slot.get("lifecycle_case") == "repair_reinstall"
                and current_cycle >= slot["return_cycle"]
            ):
                component_id = slot["original_component_id"]  # SAME physical part
                installation_id = slot["return_installation_id"]  # NEW episode
                install_cycle = slot["return_cycle"]
                age = current_cycle - install_cycle
                wear = 0.08 + 0.009 * (age / CYCLES_PER_SNAPSHOT)
            else:
                component_id = slot["secondary_component_id"]
                installation_id = slot["secondary_installation_id"]
                install_cycle = slot["event_cycle"]
                age = current_cycle - install_cycle
                wear = 0.07 + 0.011 * (age / CYCLES_PER_SNAPSHOT)

            if installation_id != current_installation:
                # Sensor/fault and maintenance history from the removed part
                # must NOT carry over into the new installation.
                current_installation = installation_id
                fault_history.clear()
                maintenance_history.clear()
                last_maintenance_cycle = None
                if installation_id == slot.get("return_installation_id"):
                    # Repaired part has undergone maintenance before its return.
                    maintenance_history.append((date, 1))
                    last_maintenance_cycle = install_cycle

            since_install = current_cycle - install_cycle
            wear = max(0.01, wear + rng.gauss(0, 0.025))
            type_offsets = {
                "pump": 0.25, "valve": 0.12, "actuator": 0.20,
                "sensor": 0.08, "compressor": 0.35,
            }
            offset = type_offsets[slot["component_type"]]
            vibration = 1.20 + 1.55 * wear + offset + rng.gauss(0, 0.25)
            temperature = 48.0 + 16.0 * wear + 2 * offset + rng.gauss(0, 3.3)
            pressure = -0.25 + 1.90 * wear + rng.gauss(0, 0.8)
            if rng.random() < 0.005:
                vibration += rng.uniform(1, 2.5)
            if rng.random() < 0.005:
                temperature += rng.uniform(10, 22)
            if rng.random() < 0.005:
                pressure += rng.uniform(3, 6)

            weekly_faults = poisson(rng, 0.05 + 1.70 * max(0.0, wear - 0.19) ** 1.7)
            weekly_maintenance = int(rng.random() < min(0.28, 0.03 + wear * 0.10))
            fault_history.append((date, weekly_faults))
            maintenance_history.append((date, weekly_maintenance))
            fault_count_30d = sum(
                count for d, count in fault_history if (date - d).days < 30
            )
            maintenance_count_90d = sum(
                count for d, count in maintenance_history if (date - d).days < 90
            )
            if weekly_maintenance:
                last_maintenance_cycle = current_cycle
            since_maintenance = (
                current_cycle - last_maintenance_cycle
                if last_maintenance_cycle is not None else ""
            )
            if fault_count_30d == 0:
                fault_code = "NONE"
            elif fault_count_30d >= 3 and wear > 0.62:
                fault_code = rng.choice(["F-ENGINE", "F-VIBRATION", "F-PRESSURE"])
            else:
                fault_code = rng.choice(["F-SENSOR", "F-INSPECTION", "F-VIBRATION"])

            removal_cycle = unscheduled_by_install.get(installation_id)
            label = int(
                removal_cycle is not None
                and 0 < removal_cycle - current_cycle <= HORIZON_CYCLES
            )

            # When would this outcome actually be KNOWN? For a positive,
            # at the unscheduled removal; for a negative, after 30 cycles
            # of follow-up (or earlier at a scheduled component removal).
            horizon_end = current_cycle + HORIZON_CYCLES
            ending_event = removal_by_install.get(installation_id)
            if label:
                known_cycle = removal_cycle
            elif (
                ending_event is not None
                and ending_event["removal_type"] == "SCHEDULED"
                and current_cycle < ending_event["removal_cycle"] <= horizon_end
            ):
                known_cycle = ending_event["removal_cycle"]
            else:
                known_cycle = horizon_end
            label_available_at = date_for_cycle(slot, known_cycle)

            records.append({
                "record_id": 0,
                "snapshot_date": date.date().isoformat(),
                "aircraft_id": slot["aircraft_id"],
                "aircraft_model": slot["aircraft_model"],
                "component_id": component_id,
                "installation_id": installation_id,
                "component_type": slot["component_type"],
                "component_position": slot["position"],
                "flight_cycles": current_cycle,
                "cycles_since_install": since_install,
                "vibration_rms": blank_with_probability(rng, round(max(0.0, vibration), 4), 0.04),
                "temperature_c": blank_with_probability(rng, round(temperature, 3), 0.03),
                "pressure_delta": blank_with_probability(rng, round(pressure, 4), 0.04),
                "fault_count_30d": fault_count_30d,
                "maintenance_count_90d": maintenance_count_90d,
                "last_maintenance_cycles_ago": since_maintenance,
                "fault_code": blank_with_probability(rng, fault_code, 0.01),
                "unscheduled_removal_next_30_cycles": label,
                "label_available_at": label_available_at.isoformat(timespec="seconds"),
                "split": slot["split"],
            })

    records.sort(key=lambda r: (
        r["snapshot_date"], r["aircraft_id"], r["component_position"]
    ))
    for i, r in enumerate(records, start=1):
        r["record_id"] = i

    # --- Strict validation ---
    expected_total = N_AIRCRAFT * POSITIONS_PER_AIRCRAFT * SNAPSHOTS_PER_POSITION
    assert len(records) == expected_total == 10_000
    assert len({r["aircraft_id"] for r in records}) == N_AIRCRAFT
    assert len(unscheduled_by_install) == sum(SPLIT_EVENTS.values()) == 100
    assert sum(r["unscheduled_removal_next_30_cycles"] for r in records) == 200

    for row in records:
        cycle = int(row["flight_cycles"])
        inst = row["installation_id"]
        ending = removal_by_install.get(inst)
        if ending is not None:
            assert cycle < int(ending["removal_cycle"]), "Removed installation persisted after removal"
        assert datetime.fromisoformat(row["label_available_at"]) >= datetime.fromisoformat(row["snapshot_date"])
        original_event = unscheduled_by_install.get(inst)
        expected_label = int(
            original_event is not None and 0 < original_event - cycle <= HORIZON_CYCLES
        )
        assert row["unscheduled_removal_next_30_cycles"] == expected_label

    # A repaired original can recur ONLY under a different installation_id.
    reinstall_slots = [s for s in slots if s.get("lifecycle_case") == "repair_reinstall"]
    replace_slots = [s for s in slots if s.get("lifecycle_case") == "replacement"]
    for slot in reinstall_slots:
        component_installations = {
            r["installation_id"] for r in records
            if r["component_id"] == slot["original_component_id"]
        }
        assert component_installations == {
            slot["original_installation_id"], slot["return_installation_id"]
        }
    for slot in replace_slots:
        assert not any(
            r["component_id"] == slot["original_component_id"]
            and r["flight_cycles"] > slot["event_cycle"]
            for r in records
        )
    for slot in slots:
        assert all(
            r["split"] == slot["split"]
            for r in records if r["aircraft_id"] == slot["aircraft_id"]
        )
    ranges = {
        split: (
            min(r["snapshot_date"] for r in records if r["split"] == split),
            max(r["snapshot_date"] for r in records if r["split"] == split),
        ) for split in SPLIT_AIRCRAFT
    }
    assert ranges["train"][1] < ranges["validation"][0]
    assert ranges["validation"][1] < ranges["test"][0]
    keys = [(r["snapshot_date"], r["aircraft_id"], r["component_position"]) for r in records]
    assert len(keys) == len(set(keys)), "Duplicate aircraft/position/day snapshots"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "synthetic_mro_10000.csv"
    removal_path = out_dir / "removal_events_AUDIT_ONLY.csv"
    installation_path = out_dir / "installation_events_AUDIT_ONLY.csv"
    write_csv(dataset_path, COLUMNS, records)
    write_csv(removal_path, EVENT_COLUMNS, sorted(removal_events, key=lambda e: (
        e["removal_date"], e["aircraft_id"], e["component_position"]
    )))
    write_csv(installation_path, INSTALL_COLUMNS, sorted(installation_events, key=lambda e: (
        e["split"], e["aircraft_id"], e["component_position"], e["installation_cycle"]
    )))

    print(f"Dataset:       {dataset_path}")
    print(f"Removal audit: {removal_path} (NEVER use as model features)")
    print(f"Install audit: {installation_path} (NEVER use as model features)")
    print(f"Rows: {len(records):,} | Positives: {sum(r['unscheduled_removal_next_30_cycles'] for r in records):,} (2.00%)")
    print(f"UNSCHEDULED removals: {len(unscheduled_by_install)}")
    print(f"Scenario counts: replacement={len(replace_slots)}, repaired/reinstalled={len(reinstall_slots)}")
    for split in SPLIT_AIRCRAFT:
        subset = [r for r in records if r["split"] == split]
        n_positive = sum(r["unscheduled_removal_next_30_cycles"] for r in subset)
        print(f"{split:<11}: {len(subset):5,} rows | {n_positive:3} positives ({n_positive / len(subset):.2%})")

    return dataset_path, removal_path, installation_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    make_dataset(seed=args.seed, out_dir=args.output_dir)
