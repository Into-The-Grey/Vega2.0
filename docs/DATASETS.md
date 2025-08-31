# Datasets

[Back to Docs Hub](README.md)

This document explains dataset formats, preparation, and sample inputs.

## Format

Training and evaluation data are represented as JSON Lines (JSONL), one record per line:

```json
{ "prompt": "...", "response": "..." }
```

- `prompt`: the user input or instruction
- `response`: the expected model output

## Building a dataset

Use the CLI or Python API to build `datasets/output.jsonl` from an input directory.

CLI:

```bash
python -m cli dataset build ./datasets/samples
```

Python:

```python
from datasets.prepare_dataset import build_dataset
out_path = build_dataset("./datasets/samples")
print(out_path)
```

The builder traverses the input directory and calls format-specific loaders:

- `.txt`: `load_txt` yields `(line, "")` for every non-empty line
- `.md`: `load_md` treats headings as prompts; following text is the response
- `.json`: `load_json_pairs` expects a list of `{prompt,response}` or `{ "data": [...] }`

Output: `datasets/output.jsonl`

## Sample files

- `datasets/samples/example.txt`: each non-empty line becomes a prompt
- `datasets/samples/example.md`: headings become prompts, text becomes response
- `datasets/samples/example.json`: array of objects with `prompt` and `response`

## Curation

Use the learning tools to curate data from the conversation log:

```bash
python -m cli learn curate --min-rating 4 --reviewed-only --out-path datasets/curated.jsonl
```

This exports high-quality rows from the SQLite `conversations` table based on rating and reviewed flags.

## Evaluation sets

Provide a JSONL file with expected responses and use the evaluation utility:

```bash
python -m cli learn evaluate datasets/curated.test.jsonl
```

## Notes

- Keep prompts and responses concise and relevant to your target tasks.
- Consider cleaning and deduplicating inputs to avoid overfitting.
- For larger datasets, stream processing or HF Datasets may be preferable.

