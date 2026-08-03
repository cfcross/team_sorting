"""Load and validate the installed Interface v1 machine contract.

This module is intentionally ROS-independent and is not imported by the control
path.  It provides a strict, read-only contract for tests and future tooling.
"""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any, Mapping

from .interfaces import ACTION_NAMES, JOINT_NAMES


SCHEMA_NAME = "team_sorting.interface"
SCHEMA_VERSION = 1
_RELATIVE_PATH = Path("config") / "contracts" / "interface_v1.json"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Interface contract contains invalid JSON constant: {value}")


def _ament_contract_candidate() -> Path | None:
    """Return the ament-index candidate without making ROS a hard dependency."""

    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("team_sorting")) / _RELATIVE_PATH
    except (ImportError, LookupError):
        return None


def _installed_share_candidates(module_path: Path) -> tuple[Path, ...]:
    """Derive adjacent prefix share paths from an installed module location.

    ``pip --prefix`` does not change the running interpreter's ``sys.prefix``.
    Walking the finite lexical ancestors handles both ``prefix/lib/...`` and
    ``prefix/local/lib/...`` without recursively scanning any directory.  The
    filesystem root itself is excluded so this never probes a global ``/share``.
    """

    resolved = module_path.resolve()
    return tuple(
        ancestor / "share" / "team_sorting" / _RELATIVE_PATH
        for ancestor in resolved.parents
        if ancestor.parent != ancestor
    )


def _candidate_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    ament_candidate = _ament_contract_candidate()
    if ament_candidate is not None:
        candidates.append(ament_candidate)
    module_path = Path(__file__).resolve()
    candidates.append(module_path.parents[1] / _RELATIVE_PATH)
    candidates.extend(_installed_share_candidates(module_path))
    candidates.extend(
        (
            Path(sys.prefix) / "share" / "team_sorting" / _RELATIVE_PATH,
        )
    )
    return tuple(dict.fromkeys(candidates))


def interface_contract_path() -> Path:
    """Return the first readable source-tree or installed contract path."""

    candidates = _candidate_paths()
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Interface v1 contract not found; searched: {searched}")


def _strict_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer and not bool")
    return value


def _finite_or_null(value: object, status: object, label: str) -> None:
    if value is None:
        if status not in {"unresolved", "partially_verified"}:
            raise ValueError(f"{label} null requires unresolved status")
        return
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number or explicit unresolved null")


def validate_interface_contract(payload: Mapping[str, Any]) -> None:
    """Validate the stable identity, ordering, ranges, and publisher groups."""

    if not isinstance(payload, Mapping):
        raise ValueError("Interface contract root must be an object")
    if payload.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"schema_name must be {SCHEMA_NAME}")
    if _strict_int(payload.get("schema_version"), "schema_version") != SCHEMA_VERSION:
        raise ValueError("schema_version must be exactly 1")

    action_contract = payload.get("action_contract")
    if not isinstance(action_contract, Mapping):
        raise ValueError("action_contract must be an object")
    dimensions = action_contract.get("dimensions")
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 19:
        raise ValueError("action dimensions must contain exactly 19 entries")
    if tuple(_strict_int(item.get("index"), "action index") for item in dimensions) != tuple(range(19)):
        raise ValueError("action indices must be exactly 0..18")
    if tuple(item.get("name") for item in dimensions) != ACTION_NAMES:
        raise ValueError("action names must exactly match ACTION_NAMES")
    if tuple(item.get("joint_state_name") for item in dimensions[2:]) != JOINT_NAMES:
        raise ValueError("Action 2..18 semantic mapping must exactly match JOINT_NAMES")
    for item in dimensions:
        status = item.get("status")
        for key in ("safe_min", "safe_max", "runtime_min", "runtime_max"):
            range_status = item.get("runtime_range_status", status) if key.startswith("runtime_") else status
            _finite_or_null(item.get(key), range_status, f"action[{item.get('index')}].{key}")

    joint_contract = payload.get("joint_state_contract")
    if not isinstance(joint_contract, Mapping) or tuple(joint_contract.get("names", ())) != JOINT_NAMES:
        raise ValueError("joint state names must exactly match JOINT_NAMES")

    topics = payload.get("official_control_topics")
    groups = topics.get("groups") if isinstance(topics, Mapping) else None
    expected = (
        ("base", 2, (0, 2)),
        ("spine", 1, (2, 3)),
        ("head", 2, (3, 5)),
        ("left_arm", 7, (5, 12)),
        ("right_arm", 7, (12, 19)),
    )
    if not isinstance(groups, (list, tuple)) or tuple(
        (
            group.get("group"),
            _strict_int(group.get("payload_width"), "payload_width"),
            tuple(group.get("action_slice", ())),
        )
        for group in groups
    ) != expected:
        raise ValueError("official topic groups, widths, or slices do not match v1")

    public = payload.get("public_interfaces")
    items = public.get("items") if isinstance(public, Mapping) else None
    if not isinstance(items, (list, tuple)):
        raise ValueError("public_interfaces.items must be an array")
    for item in items:
        if item.get("status") == "frozen":
            version = item.get("version")
            if type(version) is not int or version < 1:
                raise ValueError("every frozen public interface requires a positive integer version")

    unresolved = payload.get("unresolved")
    if not isinstance(unresolved, (list, tuple)) or not unresolved:
        raise ValueError("unresolved must be a non-empty array")
    for item in unresolved:
        if item.get("status") != "unresolved" or not item.get("required_test") or not item.get("must_not_assume"):
            raise ValueError("each unresolved item requires status, required_test, and must_not_assume")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_interface_contract(path: str | Path | None = None) -> Mapping[str, Any]:
    """Load, validate, and deeply freeze Interface v1.

    A fresh immutable object graph is returned for every call, so callers cannot
    mutate shared state or a module-level cache.
    """

    contract_path = Path(path) if path is not None else interface_contract_path()
    payload = json.loads(
        contract_path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    validate_interface_contract(payload)
    return _freeze(payload)
