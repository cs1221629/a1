import argparse
from pathlib import Path
import zipfile

parser = argparse.ArgumentParser(description="Create the Moodle submission ZIP.")
parser.add_argument("student_id", help="Registration number used as the ZIP root directory.")
parser.add_argument("--output", type=Path, help="Output ZIP path (defaults to <student_id>.zip).")
args = parser.parse_args()
student_id = args.student_id
zip_filename = args.output or Path(f"{student_id}.zip")

# Required files/folders as per assignment description
# Note: we explicitly exclude data/full/ and .git, .venv etc.
includes = [
    "assignment1.tex",
    "conftest.py",
    "data/README.md",
    "data/toy/",
    "Dockerfile",
    "docs/",
    "harness/",
    "README.md",
    "requirements.txt",
    "scripts/",
    "submission/",
    "tests/"
]

def add_to_zip(zipf: zipfile.ZipFile, item_path: str) -> None:
    path = Path(item_path)
    if not path.exists():
        return

    paths = [path] if path.is_file() else sorted(path.rglob("*"))
    for file_path in paths:
        if not file_path.is_file():
            continue
        if "__pycache__" in file_path.parts or ".pytest_cache" in file_path.parts:
            continue
        if file_path.parts[:2] == ("data", "full"):
            continue
        archive_path = (Path(student_id) / file_path).as_posix()
        zipf.write(file_path, archive_path)

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for item in includes:
        add_to_zip(zipf, item)

print(f"Created {zip_filename} successfully!")
