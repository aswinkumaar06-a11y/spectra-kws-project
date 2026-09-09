import os
import urllib.request
import tarfile
import shutil

# Define paths relative to the root project folder
URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
ARCHIVE = "dataset/raw/speech_commands.tar.gz"
TEMP_DIR = "dataset/raw/temp_speech_commands"
NEGATIVE_DIR = "dataset/raw/negative_words"
NOISE_DIR = "dataset/raw/noise"

def setup_directories():
    for d in [TEMP_DIR, NEGATIVE_DIR, NOISE_DIR]:
        os.makedirs(d, exist_ok=True)

def download_and_extract():
    print("Downloading Google Speech Commands v0.02 (~2.4GB)...")
    urllib.request.urlretrieve(URL, ARCHIVE)
    
    print("Extracting files...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(path=TEMP_DIR)

def organize_files():
    print("Routing files to your custom project structure...")
    
    # 1. Move background noise files
    noise_source = os.path.join(TEMP_DIR, "_background_noise_")
    if os.path.exists(noise_source):
        for file_name in os.listdir(noise_source):
            if file_name.endswith('.wav'):
                shutil.move(os.path.join(noise_source, file_name), os.path.join(NOISE_DIR, file_name))

    # 2. Move select words to act as hard-negatives against "SPECTRA"
    negative_classes = ['yes', 'no', 'stop', 'go', 'up', 'down', 'left', 'right', 'on', 'off']
    for word in negative_classes:
        word_source = os.path.join(TEMP_DIR, word)
        word_dest = os.path.join(NEGATIVE_DIR, word)
        if os.path.exists(word_source):
            if os.path.exists(word_dest):
                shutil.rmtree(word_dest) # Overwrite if exists
            shutil.move(word_source, word_dest)

def cleanup():
    print("Cleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    os.remove(ARCHIVE)
    print("Data successfully placed in dataset/raw/noise and dataset/raw/negative_words!")

if __name__ == "__main__":
    setup_directories()
    download_and_extract()
    organize_files()
    cleanup()