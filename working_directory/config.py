# working_directory/config.py
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Root directory relative to this config file
    BASE_DIR: Path = Path(__file__).resolve().parent

    # ==== CONFIG ====
    TRAIN_DIR: Path = (BASE_DIR / "data/mex2/train").resolve()
    TEST_DIR: Path  = (BASE_DIR / "data/mex2/test").resolve()
    N_DOCS: int = 1000
    RANDOM_SEED: Optional[int] = 42   # set None for non-reproducible sampling

    OUTPUT_JSONL: Path = (BASE_DIR / "data/mex3/corpus_1000.jsonl").resolve()
    OUTPUT_JSONL_TEST: Path = (BASE_DIR / "data/mex3/corpus_1000_test.jsonl").resolve()
    OUTPUT_CSV: Path   = (BASE_DIR / "data/mex3/corpus_1000.csv").resolve()

    class Config:
        # Allow .env files if you want to override defaults
        env_file = ".env"
        env_file_encoding = "utf-8"


# Create a singleton settings object
settings = Settings()
