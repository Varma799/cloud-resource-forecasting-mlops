import os
import yaml
import pickle


def read_yaml(file_path: str) -> dict:
    with open(file_path, "r") as yaml_file:
        return yaml.safe_load(yaml_file)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_object(file_path: str, obj) -> None:
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path: str):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)