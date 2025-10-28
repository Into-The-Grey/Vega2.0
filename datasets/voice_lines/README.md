# Voice Lines Dataset Folder

This folder stores CSV files containing voice lines used to train or evaluate Vega's voice systems (TTS/STT).

## Recommended CSV schema

Use UTF-8 encoding with a header row. Suggested columns:

- id: Unique identifier for the line (string)
- text: The transcript or text content of the line (string)
- audio_path: Relative path to the corresponding audio file, if available (string, optional)
- speaker: Speaker label or name (string, optional)
- emotion: Emotion tag (e.g., neutral, happy, sad, angry) (string, optional)
- language: ISO code (e.g., en, es, fr) (string, optional)
- dataset_split: train/val/test (string, optional)
- metadata: Freeform JSON string for additional fields (string, optional)

Example rows:

id,text,audio_path,speaker,emotion,language,dataset_split,metadata
001,Hello there and welcome!,audio/001.wav,vega,neutral,en,train,{"source":"manual"}
002,Thank you for trying Vega.,audio/002.wav,vega,neutral,en,val,{"domain":"onboarding"}

## File placement

- Place your CSV files in this directory (e.g., `voice_lines.csv`).
- If you have audio files, store them in a nearby folder, e.g., `datasets/voice_audio/`, and refer to them via `audio_path`.

## Usage notes

- Keep lines short and diverse for TTS quality; include punctuation for prosody.
- For STT testing, ensure audio files match the `text` and sampling parameters.
- Track dataset splits to support repeatable experiments.

## Integration pointers

- Voice training/inference modules live under `src/vega/voice/`.
- Custom loaders can parse this CSV and feed into your pipelines.
- If needed, create a lightweight loader (example below) to read CSVs into Python:

```python
import csv
from pathlib import Path
from typing import Iterator, Dict

def iter_voice_lines(csv_path: str) -> Iterator[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row
```

## Tips

- Normalize whitespace and remove BOMs in CSVs to avoid parsing issues.
- Validate audio paths exist before training jobs.
- Consider versioning datasets by filename or a VERSION column for reproducibility.
