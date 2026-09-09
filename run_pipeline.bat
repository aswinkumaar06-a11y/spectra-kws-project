@echo off
setlocal enabledelayedexpansion

echo ========================================================
echo Spectra KWS - Full Pipeline Runner (Hardened)
echo ========================================================

echo.
echo [1/6] Setting up Virtual Environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        goto error
    )
)
call venv\Scripts\activate
if !errorlevel! neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    goto error
)

echo Installing dependencies...
pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Failed to install requirements.
    goto error
)

echo.
echo [2/6] Preparing Raw Data (Non-destructive)...
python scripts\download_bg_data.py
if !errorlevel! neq 0 (
    echo [ERROR] Background data download failed.
    goto error
)

python scripts\generate_silence.py
if !errorlevel! neq 0 (
    echo [ERROR] Silence generation failed.
    goto error
)

python scripts\prepare_raw_positive.py
if !errorlevel! neq 0 (
    echo [ERROR] Raw positive preparation failed.
    goto error
)

echo.
echo [3/6] Splitting Data (Disjoint Speaker Groups and Noise Pools)...
python scripts\split_raw_only.py
if !errorlevel! neq 0 (
    echo [ERROR] Data splitting failed.
    goto error
)

echo.
echo [4/6] Augmenting Per Split...
python scripts\augment_per_split.py
if !errorlevel! neq 0 (
    echo [ERROR] Augmentation failed.
    goto error
)

echo.
echo [5/6] Extracting Features...
python scripts\extract_features.py
if !errorlevel! neq 0 (
    echo [ERROR] Feature extraction failed.
    goto error
)

echo.
echo [6/6] Training Model (Dynamic Class Balancing and Calibration)...
python scripts\train_model.py
if !errorlevel! neq 0 (
    echo [ERROR] Model training failed.
    goto error
)

echo.
echo [7/6] Evaluating INT8 Quantized TFLite Model...
python scripts\evaluate_tflite.py
if !errorlevel! neq 0 (
    echo [ERROR] TFLite evaluation failed.
    goto error
)

echo.
echo ========================================================
echo [SUCCESS] Pipeline completed successfully!
echo To test on files run: python scripts\test_spectra_model.py --mode files
echo To test live run:     python scripts\test_spectra_model.py --mode live
echo ========================================================
goto end

:error
echo.
echo ========================================================
echo [PIPELINE FAILED] See error messages above.
echo ========================================================
exit /b 1

:end
pause
