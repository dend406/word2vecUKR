import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
import gensim
from gensim.models import KeyedVectors


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fiction.lowercased.tokenized.word2vec.300d"
RESULTS_DIR = ROOT / "research_results"

GROUPS = {
    "physics": ["сила", "енергія", "швидкість", "рух", "прискорення"],
    "technology": ["комп'ютер", "робот", "програма", "процесор", "мережа"],
    "education": ["учень", "учитель", "школа", "знання", "навчання"],
    "nature": ["вода", "повітря", "сонце", "земля", "рослина"],
}


def write_similarity_csv(name: str, words: list[str], matrix: np.ndarray) -> Path:
    path = RESULTS_DIR / f"{name}_similarity.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["слово", *words])
        for word, row in zip(words, matrix):
            writer.writerow([word, *(float(value) for value in row)])
    return path


def write_heatmap(name: str, words: list[str], matrix: np.ndarray) -> Path:
    path = RESULTS_DIR / f"{name}_heatmap.png"
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="viridis", vmin=-1, vmax=1)
    axis.set_xticks(range(len(words)), labels=words, rotation=45, ha="right")
    axis.set_yticks(range(len(words)), labels=words)
    axis.set_title(name)
    figure.colorbar(image, ax=axis, label="cosine similarity")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def print_table(title: str, words: list[str], matrix: np.ndarray) -> None:
    print(f"\n{title}")
    header = f"{'слово':>14} | " + " | ".join(f"{word:>14}" for word in words)
    print(header)
    print("-" * len(header))
    for word, row in zip(words, matrix):
        print(f"{word:>14} | " + " | ".join(f"{value:14.4f}" for value in row))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    environment = {
        "sys_version": sys.version,
        "sys_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "gensim_version": gensim.__version__,
    }
    print("СЕРЕДОВИЩЕ")
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    active_venv = str(Path(sys.executable).resolve()).startswith(
        str((ROOT / ".venv").resolve())
    )
    print(f"Активне середовище .venv: {active_venv}")

    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    norms = np.linalg.norm(model.vectors, axis=1)

    print("\nМОДЕЛЬ")
    print(f"Кількість слів: {len(model.key_to_index)}")
    print(f"Розмірність векторів: {model.vector_size}")
    print(f"Середня норма: {norms.mean():.6f}")
    print(f"Мінімальна норма: {norms.min():.6f}")
    print(f"Максимальна норма: {norms.max():.6f}")

    presence = {}
    for category, words in GROUPS.items():
        present = [word for word in words if word in model.key_to_index]
        missing = [word for word in words if word not in model.key_to_index]
        presence[category] = {"present": present, "missing": missing}
        print(f"\nКатегорія {category}:")
        print("Наявні:", ", ".join(present) if present else "немає")
        print("Відсутні:", ", ".join(missing) if missing else "немає")

    created_files = []
    for category, words in GROUPS.items():
        available_words = presence[category]["present"]
        if len(available_words) < 2:
            print(f"\nМатрицю {category} не побудовано: менше двох наявних слів.")
            continue
        vectors = model[available_words]
        vectors = np.asarray(vectors, dtype=np.float64)
        normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        similarity = normalized @ normalized.T
        print_table(category, available_words, similarity)
        created_files.append(write_similarity_csv(category, available_words, similarity))
        created_files.append(write_heatmap(category, available_words, similarity))

    report = {
        "environment": environment,
        "active_venv": active_venv,
        "model": {
            "path": str(MODEL_PATH),
            "word_count": len(model.key_to_index),
            "vector_size": model.vector_size,
            "mean_norm": float(norms.mean()),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
        },
        "categories": presence,
        "created_files": [str(path) for path in created_files],
    }
    report_path = RESULTS_DIR / "research_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nЗвіт JSON: {report_path}")
    print("\nСтворені файли:")
    for path in created_files:
        print(path)


if __name__ == "__main__":
    main()
