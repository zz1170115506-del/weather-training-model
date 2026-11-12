"""Export and verify a data-free Task1 code package; never train a model.

Only the explicit runtime allowlist and one selected course-only configuration
are copied. Verification uses 128 generated records with a separate tiny
configuration; it neither reads course data nor relaxes the production hash pin.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path("configs/task1/seventh_run_course_only.json")
MODULES = (
    "__init__", "artifacts", "bagged_hgb", "config", "crossfit_weights",
    "data", "diagnostics", "ensemble", "features", "models", "ovr_hgb",
    "pipeline", "plots", "progress", "results", "search", "submission", "validation",
)
RUNTIME_FILES = (
    "code/run_task1.py", "code/verify_task1_release.py",
    "code/weather_models.py", "code/generative_weather.py",
    "code/requirements.txt", "code/requirements-boosting.txt",
    "code/requirements-round7.txt", "code/requirements-lock.txt", "code/README.txt",
) + tuple(f"code/task1/{name}.py" for name in MODULES)
MANIFEST = "release_manifest.json"


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("Release file paths must be safe relative POSIX paths.")
    return name


def _private_values(value):
    if isinstance(value, dict):
        sensitive = {"password", "token", "api_key", "secret", "username", "user"}
        for key, item in value.items():
            if str(key).lower() in sensitive:
                raise ValueError("The selected config contains a credential/account field.")
            _private_values(item)
    elif isinstance(value, list):
        for item in value:
            _private_values(item)
    elif isinstance(value, str):
        if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith(("/", "\\\\")):
            raise ValueError("The selected config contains a machine-specific absolute path.")


def _config_file(config):
    path = Path(config)
    return path if path.is_absolute() else ROOT / path


def _resolved_config(path):
    # Import only for schema validation; model factories are never fitted.
    from task1.config import resolve_config
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _private_values(raw)
    return resolve_config(raw)


def _readme(config_name):
    return f"""# Task1 course-only code release

This package contains runnable source and one selected configuration. It does
not contain course/external data, fitted models, predictions, old experiments,
personal notes, credentials or a virtual environment. Supply the three original
course CSVs unchanged: train.csv, test.csv, sample_submission.csv.

Use Python 3.12. From this package root, install the locked project dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r code/requirements-round7.txt
```

Set the directory containing the original course files. It can be anywhere;
`--data-dir` is explicit so no dataset is assumed to be bundled:

```powershell
$taskData = Read-Host 'Directory containing the original three course CSVs'
.\\.venv\\Scripts\\python.exe -X utf8 -B code/run_task1.py check --config {config_name} --data-dir $taskData
```

`check` validates the configured source hashes and dependencies without training.
Only run the following command manually when ready to train:

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -B code/run_task1.py train --config {config_name} --data-dir $taskData --artifact-root ./local_artifacts/task1
.\\.venv\\Scripts\\python.exe -X utf8 -B code/run_task1.py results --artifact-root ./local_artifacts/task1
```

Keep this package outside another Git checkout; a dirty parent checkout can
trigger the intentional clean-source training guard. On Linux/macOS substitute
`.venv/bin/python` for the Windows interpreter path. No activation is needed.
Training outputs use new run directories and are not part of this code release.
Use the saved `predict --bundle ... --output ... --data-dir ...` command path
to replay a locally trained bundle; source-assisted legacy bundles are refused.
Kaggle accepts integer labels 1..4, 3300 predictions plus the id,label header.
Upload a validated candidate manually; development CV is not a platform score.

Verify the package manifest, imports, CLI and a separate generated 128-row schema
check (no training, no course data required):

```powershell
.\\.venv\\Scripts\\python.exe -X utf8 -B code/verify_task1_release.py --verify-only --output .
```

SHA256 checks prove file consistency, not authenticity of an untrusted archive.
The manifest records the exact selected config and copied runtime files.
See code/README.txt for module responsibilities and experiment provenance.
"""


def _run(command, cwd, *, input_text=None):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # Do not let an unrelated local PYTHONPATH hide a missing packaged module.
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(command, cwd=cwd, env=environment, input=input_text,
                            text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError("Release verification failed: " + result.stderr[-3000:])
    return result.stdout


def _synthetic_inputs(directory):
    numeric = ["Temperature", "Humidity", "Wind Speed", "Precipitation (%)",
               "Atmospheric Pressure", "UV Index", "Visibility (km)"]
    categorical = ["Cloud Cover", "Season", "Location"]
    features = numeric + categorical
    rows = []
    for label in range(1, 5):
        for offset in range(32):
            rows.append([100 * label + offset, 12.0 * label + offset / 100,
                         40 + offset, 1.0 + offset / 10, 10.0 * label + offset / 10,
                         1000 + label, label, 10 - label,
                         ("clear", "cloudy", "overcast", "partly cloudy")[label - 1],
                         ("Summer", "Autumn", "Spring", "Winter")[offset % 4],
                         ("coastal", "inland", "mountain")[offset % 3], label])
    test = [[10001 + i] + row[1:-1] for i, row in enumerate(rows[:8])]
    for filename, header, values in (
        ("train.csv", ["id", *features, "label"], rows),
        ("test.csv", ["id", *features], test),
        ("sample_submission.csv", ["id", "label"], [[row[0], 1] for row in reversed(test)]),
    ):
        with (directory / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(values)


@contextmanager
def _scratch_directory(parent):
    # Keep generated fixtures beside the release, outside its immutable code
    # tree. Inheriting the parent's ACL also works in restricted Windows setups
    # where a process cannot write to the system's global Temp directory.
    parent = Path(parent).resolve()
    scratch = parent / ("task1_release_check_" + uuid.uuid4().hex)
    scratch.mkdir()
    try:
        yield scratch
    finally:
        target = scratch.resolve()
        if target.parent != parent or not target.name.startswith("task1_release_check_"):
            raise RuntimeError("Refusing unsafe generated-fixture cleanup.")
        shutil.rmtree(target)


def verify_release(output):
    """Read a release manifest and run non-training checks in an isolated cwd."""
    root = Path(output).resolve()
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    config_name = _safe_relative(manifest["config_file"])
    expected_names = {*RUNTIME_FILES, "README.md", config_name}
    if (manifest.get("schema_version") != 1 or manifest.get("training_data_policy") != "course_train_only"
            or set(manifest["files_sha256"]) != expected_names):
        raise ValueError("Release manifest does not match the explicit runtime allowlist.")
    for name, expected in manifest["files_sha256"].items():
        _safe_relative(name)
        path = root / name
        if (not path.resolve().is_relative_to(root) or path.is_symlink()
                or not path.is_file() or _digest(path) != expected):
            raise ValueError(f"Missing or changed release file: {name}")
    config_prefix = PurePosixPath(config_name).parts
    if (len(config_prefix) != 3 or config_prefix[:2] != ("configs", "task1")
            or not config_name.endswith(".json")):
        raise ValueError("Release requires one config under configs/task1/.")
    # Reject unexpected files only at export; later virtual environments and
    # training artifacts do not invalidate the immutable code allowlist.
    with _scratch_directory(root.parent) as scratch:
        _synthetic_inputs(scratch)
        fixture_config = {"schema_version": 1,
            "description": "Generated release-verification fixture, not course results.",
            "models": [{"name": "release_check_only", "model_id": "numpy_cart",
                        "model_params": {"max_depth": 2}}]}
        fixture = scratch / "synthetic_config.json"
        fixture.write_text(json.dumps(fixture_config), encoding="utf-8")
        script = root / "code/run_task1.py"
        help_output = _run([sys.executable, "-X", "utf8", "-B", str(script), "--help"], scratch)
        if "validate-submission" not in help_output:
            raise RuntimeError("Packaged CLI help is incomplete.")
        check = json.loads(_run([sys.executable, "-X", "utf8", "-B", str(script),
            "check", "--config", str(fixture), "--data-dir", str(scratch)], scratch))
        if (check.get("status") != "ready" or check.get("training_started") is not False
                or check.get("data_rows", {}).get("train") != 128):
            raise RuntimeError("Generated-record check did not prove a non-training schema check.")
        # Construct the selected production models without loading any data or
        # fitting. This catches missing optional imports and packaging mistakes.
        bootstrap = "\n".join([
            "import json,sys", f"sys.path.insert(0,{str(root / 'code')!r})",
            "from task1.config import load_config", "from task1.models import make_model",
            f"config=load_config({str(root / config_name)!r})",
            "for candidate in config['models']: make_model(candidate, config['validation']['model_seed'])",
            "print(json.dumps({'production_model_factories':len(config['models']),'training_started':False}))",
        ])
        production = json.loads(_run([sys.executable, "-X", "utf8", "-B", "-"], scratch, input_text=bootstrap))
    return {"file_hashes_verified": len(expected_names), "cli_help_verified": True,
            "synthetic_training_rows": 128, "synthetic_check_status": check["status"],
            "production_model_factories": production["production_model_factories"],
            "training_started": False, "course_data_read": False}


def export_release(output, config=DEFAULT_CONFIG):
    """Create a fresh data-free directory from exact source-file allowlists."""
    destination = Path(output).resolve()
    if destination.exists() or Path(output).is_symlink():
        raise FileExistsError("Release output must not exist; previous releases are never overwritten.")
    config_path = _config_file(config).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*\.json", config_path.name):
        raise ValueError("The selected config needs a safe JSON filename.")
    _resolved_config(config_path)
    selected_name = "configs/task1/" + config_path.name
    copies = {name: ROOT / name for name in RUNTIME_FILES} | {selected_name: config_path}
    for name, source in copies.items():
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Required release source is missing or linked: {name}")
        if name in RUNTIME_FILES and not source.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Runtime source escapes the checkout: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, source in copies.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
    (destination / "README.md").write_text(_readme(selected_name), encoding="utf-8")
    manifest = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "training_data_policy": "course_train_only", "config_file": selected_name,
                "files_sha256": {name: _digest(destination / name)
                                 for name in sorted([*copies, "README.md"])},
                "scope": "Code and one course-only config only; no data, models, predictions, old configs, results, personal documents or environment."}
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    result = verify_release(destination)
    manifest["verification"] = result
    (destination / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(destination), "config_file": selected_name, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh directory for export, or an existing package with --verify-only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true", help="Check a saved manifest and non-training CLI checks.")
    args = parser.parse_args()
    result = verify_release(args.output) if args.verify_only else export_release(args.output, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
