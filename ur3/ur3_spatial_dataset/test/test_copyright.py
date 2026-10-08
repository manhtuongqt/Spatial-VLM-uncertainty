from pathlib import Path


def test_source_files_have_no_tabs():
    root = Path(__file__).parents[1]
    for path in (root / "ur3_spatial_dataset").glob("*.py"):
        assert "\t" not in path.read_text(encoding="utf-8")
