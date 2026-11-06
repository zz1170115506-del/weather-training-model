"""Small, explicit experiment records; large artifacts stay outside Git."""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import json
import os
import platform
import subprocess
import uuid

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT.parent / "local_artifacts" / "task1"


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"), default=json_default)


def content_hash(value):
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=json_default) + "\n", encoding="utf-8")


def git_state():
    def git(*args):
        try:
            result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
        except OSError:
            return None
        if result.returncode:
            return None
        return result.stdout.strip()
    commit = git("rev-parse", "HEAD")
    return {"available": commit is not None, "commit": commit,
            "dirty": bool(git("status", "--porcelain")),
            "branch": git("branch", "--show-current")}


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib",
                 "matplotlib", "threadpoolctl", "catboost"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    from threadpoolctl import threadpool_info
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": cpu, "logical_cpu_count": os.cpu_count(), "servers": 1,
            "gpu": "not used", "packages": packages,
            "threadpools": [{k: entry.get(k) for k in
                              ("user_api", "internal_api", "num_threads", "version")}
                             for entry in threadpool_info()]}


def new_run(stage, config, artifact_root=None):
    root = Path(artifact_root or DEFAULT_ARTIFACT_ROOT).resolve()
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%SZ}_{stage}_{content_hash(config)[:8]}_{uuid.uuid4().hex[:6]}"
    out = root / run_id
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "resolved_config.json", config)
    save_json(out / "environment.json", environment())
    save_json(out / "run.json", {"run_id": run_id, "stage": stage,
                                "timestamp_utc": now.isoformat(), "git": git_state(),
                                "config_sha256": content_hash(config), "status": "running"})
    return out


def finish_run(out, metadata):
    out = Path(out)
    current = json.loads((out / "run.json").read_text(encoding="utf-8"))
    current.update(metadata)
    current["status"] = "complete"
    current["files_sha256"] = {
        str(p.relative_to(out)).replace("\\", "/"): file_hash(p)
        for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run.json"
    }
    save_json(out / "run.json", current)
