import os
import pickle
from typing import Any


def save_object(file_path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: str) -> Any:
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)
