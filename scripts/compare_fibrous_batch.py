import argparse
import csv
import json
from pathlib import Path

from contact_angle.fibrous import measure_fibrous_bytes


def strict_pass(left_error, right_error, tolerance=5.0):
    return abs(left_error) < tolerance and abs(right_error) < tolerance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("Choose an empty output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    before = json.loads(args.before.read_text())
    rows = []
    all_methods = []
    for old in before:
        no = old["id"]
        raw = (args.source / old["filename"]).read_bytes()
        result, png = measure_fibrous_bytes(raw)
        folder = args.output / f"{no:02d}"
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (folder / "annotated.png").write_bytes(png)
        left, right = result["left"], result["right"]
        row = {k: old[k] for k in ("id", "filename", "volume_ul", "manual_left", "manual_right")}
        row.update(
            old_left=old["left"],
            old_right=old["right"],
            left=left["angle_deg"],
            right=right["angle_deg"],
            old_l_min=old["l_min"],
            old_l_max=old["l_max"],
            old_r_min=old["r_min"],
            old_r_max=old["r_max"],
            l_min=left["min_deg"],
            l_max=left["max_deg"],
            r_min=right["min_deg"],
            r_max=right["max_deg"],
            old_l_span=old["l_max"] - old["l_min"],
            old_r_span=old["r_max"] - old["r_min"],
            l_span=left["span_deg"],
            r_span=right["span_deg"],
            left_error=left["angle_deg"] - old["manual_left"],
            right_error=right["angle_deg"] - old["manual_right"],
            old_left_error=old["left"] - old["manual_left"],
            old_right_error=old["right"] - old["manual_right"],
        )
        row.update(
            l_eligible_min=left["all_eligible_min_deg"],
            l_eligible_max=left["all_eligible_max_deg"],
            r_eligible_min=right["all_eligible_min_deg"],
            r_eligible_max=right["all_eligible_max_deg"],
            left_family_count=len(left["family_angles"]),
            right_family_count=len(right["family_angles"]),
        )
        row["manual_mean"] = (row["manual_left"] + row["manual_right"]) / 2
        row["old_mean"] = (row["old_left"] + row["old_right"]) / 2
        row["mean"] = (row["left"] + row["right"]) / 2
        row["old_mean_error"] = row["old_mean"] - row["manual_mean"]
        row["mean_error"] = row["mean"] - row["manual_mean"]
        row["old_pass_both"] = strict_pass(row["old_left_error"], row["old_right_error"])
        row["pass_both"] = strict_pass(row["left_error"], row["right_error"])
        row["old_pass_mean"] = abs(row["old_mean_error"]) < 5
        row["pass_mean"] = abs(row["mean_error"]) < 5
        row["left_families"] = ",".join(left["family_angles"])
        row["right_families"] = ",".join(right["family_angles"])
        row["status"] = "pass_manual_comparison" if row["pass_both"] else "fail_manual_comparison"
        rows.append(row)
        for run in result["runs"]:
            all_methods.append({"id": no, **run})
        print(no, round(row["left"], 2), round(row["right"], 2), row["pass_both"], flush=True)
    (args.output / "records.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with (args.output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = sorted(set().union(*(r.keys() for r in all_methods)))
    with (args.output / "methods.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_methods)


if __name__ == "__main__":
    main()
