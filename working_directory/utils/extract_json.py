import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

# ==== CONFIG ====
TRAIN_DIR = Path("../data/mex2/train")                  
ALL_SECTIONS_CSV = Path("../cache/all_sections.csv")
ALL_SECTIONS_PARQUET = Path("../cache/all_sections.parquet")
SAMPLED_SECTIONS_CSV = Path("../cache/sampled_1000_sections.csv")
SAMPLED_CACHE_JSON = Path("../cache/sampled_sections_cache.json")
N_SECTIONS = 1000
FORCE_RESAMPLE = False

# ==== HELPERS ====
def list_json_files(train_dir: Path) -> List[Path]:
    return sorted([p for p in train_dir.iterdir() if p.suffix.lower() == ".json"])

def load_sections_from_file(fp: Path) -> List[Dict]:
    """
    Each file is a JSON list of objects like:
        [{"section_title": "...", "text": "..."}, ...]
    Returns a list of dicts with file_name, section_index, section_title, text.
    """
    sections_out = []
    with fp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return sections_out
    for idx, sec in enumerate(data):
        sec = sec or {}
        if sec.get("text", None) == "" or sec.get("section_title", None) == "":
            continue

        sections_out.append({
            "file_name": fp.name,
            "section_index": idx,
            "section_title": sec.get("section_title", None),
            "text": sec.get("text", None)
        })
    return sections_out

def build_all_sections_df(train_dir: Path) -> pd.DataFrame:
    rows = []
    files = list_json_files(train_dir)
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {train_dir.resolve()}")
    for fp in files:
        try:
            rows.extend(load_sections_from_file(fp))
        except Exception as e:
            print(f"[WARN] Skipping {fp.name}: {e}")
    df = pd.DataFrame(rows)
    # stable ID per section for caching
    if not df.empty:
        df["section_id"] = df["file_name"].astype(str) + ":" + df["section_index"].astype(str)
    else:
        df = pd.DataFrame(columns=["file_name","section_index","section_title","text","section_id"])
    return df

def save_df(df: pd.DataFrame, csv_path: Path, parquet_path: Path):
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)  # optional, faster reloads
    except Exception as e:
        print(f"[INFO] Could not write parquet ({parquet_path.name}): {e}")

def load_or_create_all_sections(train_dir: Path) -> pd.DataFrame:
    if ALL_SECTIONS_CSV.exists():
        df = pd.read_csv(ALL_SECTIONS_CSV)
        # ensure section_id exists (for older runs)
        if "section_id" not in df.columns:
            df["section_id"] = df["file_name"].astype(str) + ":" + df["section_index"].astype(str)
        return df
    # build and save
    df = build_all_sections_df(train_dir)
    print(df.info())
    df = df.loc[~df['text'].isnull() | ~df['section_title'].isnull()]
    print(df.info())

    save_df(df, ALL_SECTIONS_CSV, ALL_SECTIONS_PARQUET)
    return df

def load_or_sample_1000(df_all: pd.DataFrame) -> pd.DataFrame:
    import random
    # If we already have a cached sample AND not forcing resample, reuse it
    if SAMPLED_CACHE_JSON.exists() and not FORCE_RESAMPLE:
        with SAMPLED_CACHE_JSON.open("r", encoding="utf-8") as f:
            cached_ids = json.load(f)  # list of section_id
        sample_df = df_all[df_all["section_id"].isin(cached_ids)].copy()
        # In case some sections disappeared (unlikely), shrink to those still present
        return sample_df

    # Fresh sample (or forced)
    population = df_all["section_id"].tolist()
    if len(population) == 0:
        raise ValueError("No sections available to sample from.")
    k = min(N_SECTIONS, len(population))
    sampled_ids = random.sample(population, k=k)

    # cache the IDs
    with SAMPLED_CACHE_JSON.open("w", encoding="utf-8") as f:
        json.dump(sampled_ids, f, indent=2)

    sample_df = df_all[df_all["section_id"].isin(sampled_ids)].copy()
    return sample_df

# ==== MAIN FLOW ====
if __name__ == "__main__":
    df_all = load_or_create_all_sections(TRAIN_DIR)
    print(f"[INFO] All sections: {len(df_all):,} rows | Saved to {ALL_SECTIONS_CSV}")

    df_sample = load_or_sample_1000(df_all)
    df_sample.to_csv(SAMPLED_SECTIONS_CSV, index=False)
    print(f"[INFO] Sampled sections: {len(df_sample):,} rows | Saved to {SAMPLED_SECTIONS_CSV}")
