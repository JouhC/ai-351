import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# ==== CONFIG ====
from working_directory.config import settings

# ==== HELPERS ====
def list_json_files(train_dir: Path) -> List[Path]:
    files = sorted([p for p in train_dir.iterdir() if p.suffix.lower() == ".json"])
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {train_dir.resolve()}")
    return files

def load_sections_from_file(fp: Path) -> List[Dict]:
    """
    Each file is a JSON list of objects like:
        [{"section_title": "...", "text": "..."}, ...]
    Returns a list of section dicts (possibly filtered).
    """
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    out = []
    for sec in data:
        if not isinstance(sec, dict):
            continue
        text = (sec.get("text") or "").strip()
        title = (sec.get("section_title") or "").strip()
        # Keep if either text or title has content
        if text or title:
            out.append({"section_title": title, "text": text})
    return out

def build_doc_text(sections: List[Dict]) -> str:
    """
    Concatenate non-empty sections into a single document string.
    Prefer text; include title as a prefix when text is empty.
    """
    chunks: List[str] = []
    for sec in sections:
        txt = sec.get("text", "").strip()
        title = sec.get("section_title", "").strip()
        if txt:
            chunks.append(txt)
        elif title:
            chunks.append(title)
    return "\n\n".join(chunks).strip()

# ==== MAIN ====
if __name__ == "__main__":
    print("running!")
    if settings.RANDOM_SEED is not None:
        random.seed(settings.RANDOM_SEED)

    files = list_json_files(settings.TEST_DIR)
    k = min(settings.N_DOCS, len(files))
    sampled_files = random.sample(files, k=k)

    records: List[Tuple[str, str, int]] = []  # (file_name, doc_text, n_sections_used)

    for fp in sampled_files:
        try:
            sections = load_sections_from_file(fp)
            doc_text = build_doc_text(sections)
            if doc_text:  # keep only non-empty documents
                records.append((fp.name, doc_text, len(sections)))
            else:
                # skip empty docs silently; alternatively log if you want
                pass
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")

    if not records:
        raise ValueError("No non-empty documents after sampling and parsing.")

    # Save JSONL (corpus)
    #settings.OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    #with settings.OUTPUT_JSONL.open("w", encoding="utf-8") as f:
    #    for fname, text, nsec in records:
    #        f.write(json.dumps({"file_name": fname, "text": text, "n_sections": nsec}, ensure_ascii=False) + "\n")

    settings.OUTPUT_JSONL_TEST.parent.mkdir(parents=True, exist_ok=True)
    with settings.OUTPUT_JSONL_TEST.open("w", encoding="utf-8") as f:
        for fname, text, nsec in records:
            f.write(json.dumps({"file_name": fname, "text": text, "n_sections": nsec}, ensure_ascii=False) + "\n")         

    # Also save a CSV for quick inspection / Pandas loading (optional)
    #df = pd.DataFrame(records, columns=["file_name", "text", "n_sections"])
    #df.to_csv(settings.OUTPUT_CSV, index=False)

    print(f"[INFO] Sampled docs: {len(records):,} / requested {k:,}")
    print(f"[INFO] Corpus saved: {settings.OUTPUT_JSONL}")
    print(f"[INFO] CSV saved:    {settings.OUTPUT_CSV}")
