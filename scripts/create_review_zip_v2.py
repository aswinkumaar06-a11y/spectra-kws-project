import os
import shutil
import zipfile

PROJECT_ROOT = "d:/spectra-kws-project"
STAGING_DIR = os.path.join(PROJECT_ROOT, "Spectra_Arena_Review_v2")
ZIP_PATH = os.path.join(PROJECT_ROOT, "Spectra_Arena_Review_v2.zip")

if os.path.exists(STAGING_DIR):
    shutil.rmtree(STAGING_DIR)
os.makedirs(STAGING_DIR, exist_ok=True)

# 1. Scripts (exclude __pycache__)
shutil.copytree(
    os.path.join(PROJECT_ROOT, "scripts"),
    os.path.join(STAGING_DIR, "scripts"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)

# 2. Tests (exclude __pycache__)
shutil.copytree(
    os.path.join(PROJECT_ROOT, "tests"),
    os.path.join(STAGING_DIR, "tests"),
    ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
)

# 3. Models
os.makedirs(os.path.join(STAGING_DIR, "models"), exist_ok=True)
for mf in ["model_manifest.json", "spectra_model.tflite", "spectra_model.cc"]:
    src = os.path.join(PROJECT_ROOT, "models", mf)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(STAGING_DIR, "models", mf))

# 4. Logs
os.makedirs(os.path.join(STAGING_DIR, "logs"), exist_ok=True)
for lf in ["evaluate_tflite.log", "unittest_verbose.log"]:
    src = os.path.join(PROJECT_ROOT, "logs", lf)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(STAGING_DIR, "logs", lf))

# 5. Dataset splits & manifests
os.makedirs(os.path.join(STAGING_DIR, "dataset", "splits"), exist_ok=True)
for df in ["noise_pools.json", "dataset_summary.json", "train_manifest.json", "val_manifest.json", "test_manifest.json"]:
    src = os.path.join(PROJECT_ROOT, "dataset", "splits", df)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(STAGING_DIR, "dataset", "splits", df))

# 6. Root files
for rf in ["run_pipeline.bat", "requirements.txt", "README.md", "arena_handover_report.md", "walkthrough.md", "revision.txt", "CHANGELOG_v1_to_v2.txt"]:
    src = os.path.join(PROJECT_ROOT, rf)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(STAGING_DIR, rf))

# 7. Create ZIP
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(STAGING_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, PROJECT_ROOT)
            zipf.write(file_path, arcname)

# 8. Clean up staging directory
shutil.rmtree(STAGING_DIR)

print(f"Created ZIP: {ZIP_PATH}")
print(f"ZIP Size: {os.path.getsize(ZIP_PATH):,} bytes")

# 9. Verify ZIP contents
with zipfile.ZipFile(ZIP_PATH, "r") as zipf:
    print(f"\nZIP Contents ({len(zipf.namelist())} files):")
    for info in zipf.infolist():
        print(f"  {info.filename} ({info.file_size:,} bytes)")
