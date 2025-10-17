# Transcription and Diarization Script

This script transcribes audio or video files using the WhisperX library and performs speaker diarization using Pyannote.audio. It generates multiple output formats, including SRT, VTT, and a formatted Markdown file.

## Features

- **Transcription**: Utilizes the powerful Whisper models for accurate speech-to-text conversion.
- **Speaker Diarization**: Identifies and assigns different speakers to the transcribed text.
- **Multiple Output Formats**: Generates `.srt` and `.vtt` subtitle files, as well as a user-friendly Markdown (`.md`) file with speaker labels and paragraph breaks.
- **Handles Various Inputs**: Can process both audio and video files by extracting the audio stream with ffmpeg.
- **Customizable**: Allows for selection of different Whisper models, language detection, and other transcription parameters.

## Prerequisites

Before you begin, ensure you have the following installed and configured:

1. **Python 3.10+**: Make sure you have a compatible version of Python installed.
2. **ffmpeg**: The script requires `ffmpeg` to be installed and accessible in your system's PATH. You can download it from [ffmpeg.org](https://ffmpeg.org/download.html).
3. **Hugging Face Account and Token**: Speaker diarization requires a model from Hugging Face.
    - Create an account on [Hugging Face](https://huggingface.co/).
    - Generate an access token in your settings: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    - Set the token as an environment variable named `HF_TOKEN`. For example, in PowerShell:

        ```powershell
        $env:HF_TOKEN = "your_token_here"
        ```

## Installation

1. **Clone the repository or download the files.**

2. **Install the required Python packages:**

    It is highly recommended to use a virtual environment to avoid conflicts with other projects.

    ```bash
    # Create and activate a virtual environment (optional but recommended)
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

    # Install dependencies
    pip install -r requirements.txt
    ```

## Dependencies

This project relies on several key dependencies that will be automatically installed:

- **PyTorch & Torchaudio**: Deep learning framework required for audio processing and model inference
- **WhisperX**: Enhanced version of OpenAI's Whisper for speech recognition with improved timestamp alignment
- **Pyannote.audio**: Speaker diarization toolkit for identifying different speakers
- **hf-transfer**: Accelerated downloads from Hugging Face model hub
- **tqdm**: Progress bars for long-running transcription tasks
- **NumPy**: Numerical computing support

Note: PyTorch installation may take some time and requires significant disk space (several GB).

## Usage

Run the script from your terminal using the following command structure:

```bash
python transcribe.py --input <path_to_your_file> [options]
```

### Required Arguments

- `--input`: The path to the video or audio file you want to transcribe.

### Optional Arguments

- `--model`: The Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large-v2`). Default is `large-v2`.
- `--language`: The language code of the audio (e.g., `en`, `pt`). If not specified, the script will auto-detect the language.
- `--batch_size`: The number of audio segments to process in parallel during transcription. A higher value can speed up the process at the cost of higher memory usage. Default is `16`.
- `--compute_type`: The compute type for the model (e.g., `int8`, `float32`). Default is `int8`.
- `--align_model`: Path or Hugging Face repo ID for the alignment model.
- `--align_model_dir`: Directory to cache the alignment model weights.

### Example

```bash
python transcribe.py --input "my_video.mp4" --model "medium" --language "en"
```

## Output

After a successful run, the script will generate three files in the same directory as your input file:

- **`your_file.srt`**: A standard subtitle file with timestamps and speaker labels.
- **`your_file.vtt`**: A WebVTT subtitle file, suitable for web players.
- **`your_file.md`**: A Markdown file containing the full transcription, formatted with speaker headings and paragraph breaks for readability.
