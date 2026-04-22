from __future__ import annotations

from pathlib import Path

from magi.core.profiles import list_profiles, load_profile


def test_load_builtin_profile() -> None:
    profile = load_profile("security-review")

    assert profile is not None
    assert profile.name == "security-review"
    assert profile.route_mode == "decision"
    assert profile.presentation_style == "security_review"


def test_list_profiles_includes_builtin_profiles() -> None:
    profiles = list_profiles()
    names = {item.name for item in profiles}

    assert "security-review" in names
    assert "policy-triage" in names
    assert "incident-review" in names
    assert "exec-brief" in names
    assert "vendor-review" in names
    assert any(item.presentation_style == "executive_brief" for item in profiles)


def test_workspace_profile_overrides_builtin_listing(tmp_path: Path) -> None:
    custom = tmp_path / "security-review.yaml"
    custom.write_text(
        "\n".join(
            [
                "name: security-review",
                "description: Workspace override",
                "route_mode: recommend",
                "presentation_style: vendor_review",
            ]
        ),
        encoding="utf-8",
    )

    profiles = list_profiles(base_dir=tmp_path)
    selected = next(item for item in profiles if item.name == "security-review")

    assert selected.source == "workspace"
    assert selected.path == custom.resolve()
    assert selected.presentation_style == "vendor_review"
