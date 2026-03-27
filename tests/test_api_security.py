from __future__ import annotations

from pathlib import Path

import pytest

from scripts.serve_dashboard import _resolve_config_path, _to_bool, _validate_choice


def test_resolve_config_path_blocks_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path / "outside.yaml"
    outside.write_text("dataset: {}", encoding="utf-8")
    with pytest.raises(ValueError):
        _resolve_config_path(str(outside), "configs/portable_interactive.yaml")


def test_resolve_config_path_allows_same_as_service_base(tmp_path: Path) -> None:
    outside = tmp_path / "outside_base.yaml"
    outside.write_text("dataset: {}", encoding="utf-8")
    resolved = _resolve_config_path(str(outside), str(outside))
    assert Path(resolved).resolve() == outside.resolve()


def test_validate_choice_rejects_unknown_option() -> None:
    with pytest.raises(ValueError):
        _validate_choice("random_backend", ["dense", "bm25"], "backend")


def test_to_bool_parsing() -> None:
    assert _to_bool(True, default=False) is True
    assert _to_bool("true", default=False) is True
    assert _to_bool("0", default=True) is False
    assert _to_bool("bad-value", default=True) is True
