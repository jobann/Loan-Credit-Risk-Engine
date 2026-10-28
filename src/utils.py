from pathlib import Path
import joblib

def save_artifact(obj, filepath) -> None:
    """Save a model, preprocessor, or metadata to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(filepath):
    """Load a saved artifact from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)

