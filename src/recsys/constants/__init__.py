from pathlib import Path

# This file is src/recsys/constants/__init__.py: parents[0] = constants, [1] = recsys, [2] = src, [3] = project root
ROOT = Path(__file__).resolve().parents[3]

SCHEMA_FILE = ROOT / "config" / "schema.yaml"
PARAMS_FILE = ROOT / "params.yaml"

RAW_DIR = ROOT / "data" / "RAW"                      # DVC-tracked H&M files

ARTIFACT_DIR = ROOT / "artifacts"                    # pipeline outputs, one folder per stage
INGESTION_DIR = ARTIFACT_DIR / "data_ingestion"
VALIDATION_DIR = ARTIFACT_DIR / "data_validation"
SPLIT_DIR = ARTIFACT_DIR / "data_splitter"
LISTS_DIR = ARTIFACT_DIR / "list_builder"          # one subfolder per history_end_week (evaluation vs serving)
MODEL_TRAINER_DIR = ARTIFACT_DIR / "model_trainer"
MODEL_EVALUATION_DIR = ARTIFACT_DIR / "model_evaluation"
RECOMMENDATION_TABLE_DIR = ARTIFACT_DIR / "recommendation_table"   # what the API serves (built from the serving history)
