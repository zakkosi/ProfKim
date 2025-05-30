# utils/faiss_utils.py
import faiss
import numpy as np
import os
import pickle

def save_faiss_index(index, path):
    faiss.write_index(index, f"{path}.index")

def load_faiss_index(path):
    return faiss.read_index(f"{path}.index")

def save_id_mapping(mapping: dict, path):
    with open(f"{path}_ids.pkl", "wb") as f:
        pickle.dump(mapping, f)

def load_id_mapping(path):
    with open(f"{path}_ids.pkl", "rb") as f:
        return pickle.load(f)

def build_faiss_index(vectors: np.ndarray, id_list: list[str], path: str):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    save_faiss_index(index, path)
    save_id_mapping({i: id_ for i, id_ in enumerate(id_list)}, path)
    return index

def load_image_resized(image_path, size=(224, 224)):
    with Image.open(image_path).convert("RGB") as img:
        return img.resize(size)