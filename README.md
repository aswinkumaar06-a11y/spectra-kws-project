# spectra-kws-project

A keyword spotting system for detecting the wake word "SPECTRA".

## Setup and Running

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline:**
   We have a `run_pipeline.bat` that runs the full end-to-end processing pipeline, including data preparation, feature extraction, and model training.
   ```bash
   .\run_pipeline.bat
   ```

3. **Test the live microphone:**
   Once the model is trained, you can test it live with your microphone.
   ```bash
   python scripts\test_spectra_model.py --mode live --device 4
   ```

## Repository Structure

- `dataset/`: Contains raw, augmented, split audio data and generated features.
- `scripts/`: Python scripts for data processing, training, and testing.
- `models/`: Trained models (`.h5`, `.tflite`, `.cc`).