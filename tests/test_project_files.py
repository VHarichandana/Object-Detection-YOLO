from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_project_files_exist():
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / "README.md").is_file()


def test_readme_contains_internship_metadata():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "CITS9598" in readme
    assert "Datla Gnana Venkata Reddy" in readme
    assert "No. of Weeks:** 4" in readme
    assert "Object Detection (YOLO)" in readme
