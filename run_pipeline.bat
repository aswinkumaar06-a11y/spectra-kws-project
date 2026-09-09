@echo off
echo ========================================================
echo Spectra KWS - Full Pipeline Runner
echo ========================================================

echo.
echo [1/6] Setting up Virtual Environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/6] Preparing Data...
python scripts\download_bg_data.py
python scripts\generate_silence.py
python scripts\prepare_raw_positive.py

echo.
echo [3/6] Splitting Data...
python scripts\split.py

echo.
echo [4/6] Data Augmentation...
python scripts\augment.py

echo.
echo [5/6] Extracting Features...
python scripts\extract_features.py

echo.
echo [6/6] Training Model...
python scripts\train_model.py

echo.
echo ========================================================
echo Pipeline completed! You can now test the model.
echo To test on files run: python scripts\test_spectra_model.py --mode files
echo To test live run:     python scripts\test_spectra_model.py --mode live
echo ========================================================
pause
