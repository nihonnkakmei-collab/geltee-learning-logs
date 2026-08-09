# Data curation queue

`curate_dataset.py` is deliberately a **review queue**, not a web crawler and not an automatic trainer. It accepts JSONL records only from sources explicitly marked `approved: true` in a local allowlist.

Each input record needs `source_id`, `origin` (`licensed_source`, `public_domain`, or `user_owned`), and either `prompt` / `response` or a two-message conversation. The program rejects unapproved sources, likely personal data, malformed/very short data, exact duplicates, and near duplicates. It preserves source URL, licence, and fingerprint for every accepted record.

Example command (keep the allowlist and raw input local):

```powershell
python .\curate_dataset.py --allowlist .\my_allowlist.json --input .\raw_source.jsonl --out-dir .\curation-output
```

Review `curation-output\training_candidates.jsonl` and `curation-output\quarantine.jsonl`. Do not replace `GELTEE_TRAIN_DATA` with candidates until a separate holdout evaluation has been prepared and reviewed.

For sources under CC BY-SA 4.0, attribution and share-alike obligations still apply; record the exact source and license URL. This tool is not legal advice.
