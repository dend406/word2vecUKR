import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fiction.lowercased.tokenized.word2vec.300d"
PREVIOUS_RESULTS = ROOT / "results" / "experiment_02" / "results.json"
RESULTS_DIR = ROOT / "results" / "experiment_03"

SEED = 20261106
SAMPLE_SIZE = 500
PCA_DIMENSIONS = [100, 50, 20, 10, 5, 3]
RANDOM_PROJECTION_DIMENSIONS = [50, 20, 10, 5, 3]
TOP_K = 10


def normalized(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    return normalized(matrix) @ normalized(matrix).T


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def top10_overlap(original_matrix: np.ndarray, projected_matrix: np.ndarray) -> float:
    overlaps = []
    count = min(TOP_K, len(original_matrix) - 1)
    for index in range(len(original_matrix)):
        candidates = np.delete(np.arange(len(original_matrix)), index)
        original_neighbors = set(candidates[np.argsort(-original_matrix[index, candidates], kind="stable")[:count]])
        projected_neighbors = set(candidates[np.argsort(-projected_matrix[index, candidates], kind="stable")[:count]])
        overlaps.append(len(original_neighbors & projected_neighbors) / count)
    return float(np.mean(overlaps))


def evaluate(vectors: np.ndarray, reduced_vectors: np.ndarray, indices: np.ndarray = None) -> dict[str, float]:
    if indices is None:
        selected_vectors = vectors
        selected_reduced = reduced_vectors
    else:
        selected_vectors = vectors[indices]
        selected_reduced = reduced_vectors[indices]

    original_cosine = cosine_matrix(selected_vectors)
    reduced_cosine = cosine_matrix(selected_reduced)
    original_pairs = upper_triangle(original_cosine)
    reduced_pairs = upper_triangle(reduced_cosine)
    if len(original_pairs) < 2:
        correlation = 0.0
    else:
        correlation = float(spearmanr(original_pairs, reduced_pairs).statistic)
        if not np.isfinite(correlation):
            correlation = 0.0

    return {
        "mae": float(np.mean(np.abs(original_pairs - reduced_pairs))),
        "spearman": correlation,
        "top10_overlap": top10_overlap(original_cosine, reduced_cosine),
    }


def suitable_random_word(word: str, excluded: set[str]) -> bool:
    if word in excluded or len(word) < 3:
        return False
    if "_" in word or "-" in word or " " in word or "/" in word:
        return False
    return all("А" <= char <= "я" or char in "іїєґІЇЄҐ" for char in word)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "scope", "method", "dimensions", "explained_variance", "mae",
        "spearman", "top10_overlap", "sample_size",
    ]
    with path.open("w", encoding="utf-8-sig", newline="", errors="replace") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_metric_plot(path: Path, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5.5))
    for method in ("PCA", "RandomProjection"):
        method_rows = [row for row in rows if row["scope"] == "overall" and row["method"] == method]
        method_rows.sort(key=lambda row: row["dimensions"])
        if method_rows:
            plt.plot(
                [row["dimensions"] for row in method_rows],
                [row[metric] for row in method_rows],
                marker="o",
                label=method,
            )
    plt.gca().invert_xaxis()
    plt.xlabel("Кількість вимірів")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def markdown_report(config: dict, rows: list[dict], environment: dict) -> str:
    overall_pca = [row for row in rows if row["scope"] == "overall" and row["method"] == "PCA"]
    lines = [
        "# Експеримент 03: зменшення розмірності Word2Vec",
        "",
        "## Середовище",
        "",
        f"- Python: `{environment['python_version']}`",
        f"- Interpreter: `{environment['python_executable']}`",
        f"- NumPy: `{environment['numpy']}`",
        f"- SciPy: `{environment['scipy']}`",
        f"- scikit-learn: `{environment['sklearn']}`",
        "",
        "## Конфігурація вибірки",
        "",
        f"- Seed: `{config['seed']}`",
        f"- Загальний розмір вибірки: `{config['sample_size']}`",
        f"- Тематичні слова попереднього експерименту: `{config['category_word_count']}`",
        f"- Додаткові випадкові слова: `{config['random_word_count']}`",
        "- Random-додаток відбирається детерміновано з лексем словника довжиною від 3 символів, що складаються лише з кириличних літер без пробілів і розділювачів.",
        "",
        "## Загальні результати PCA",
        "",
        "| Вимірність | Explained variance | MAE подібності | Spearman | Top-10 overlap |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(overall_pca, key=lambda item: item["dimensions"], reverse=True):
        lines.append(
            f"| {row['dimensions']} | {row['explained_variance']:.6f} | {row['mae']:.6f} | "
            f"{row['spearman']:.6f} | {row['top10_overlap']:.6f} |"
        )

    lines.extend([
        "",
        "## Тематичні категорії",
        "",
        "| Категорія | Вимірність | MAE | Spearman | Top-10 overlap |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in sorted(
        (item for item in rows if item["scope"] in config["categories"] and item["method"] == "PCA"),
        key=lambda item: (item["scope"], -item["dimensions"]),
    ):
        lines.append(
            f"| {row['scope']} | {row['dimensions']} | {row['mae']:.6f} | "
            f"{row['spearman']:.6f} | {row['top10_overlap']:.6f} |"
        )

    lines.extend([
        "",
        "## Порівняння PCA та Random Projection",
        "",
        "| Вимірність | PCA MAE | RP MAE | PCA Spearman | RP Spearman | PCA Top-10 | RP Top-10 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dimensions in RANDOM_PROJECTION_DIMENSIONS:
        pca = next(row for row in rows if row["scope"] == "overall" and row["method"] == "PCA" and row["dimensions"] == dimensions)
        rp = next(row for row in rows if row["scope"] == "overall" and row["method"] == "RandomProjection" and row["dimensions"] == dimensions)
        lines.append(
            f"| {dimensions} | {pca['mae']:.6f} | {rp['mae']:.6f} | "
            f"{pca['spearman']:.6f} | {rp['spearman']:.6f} | "
            f"{pca['top10_overlap']:.6f} | {rp['top10_overlap']:.6f} |"
        )

    lines.extend([
        "",
        "## Обмеження",
        "",
        "- Explained variance — властивість PCA на вибірці, а не пряма міра збереження семантичних відношень.",
        "- Результати залежать від 500 словоформ, навчальної корпорації та конкретного словника.",
        "- Попарні косинусні значення залежні, тому метрики слід інтерпретувати описово.",
        "- Random Projection є базовою лінією і не оптимізує розподіл вибірки.",
        "- Коректного висновку про практичну придатність низьковимірних представників без зовнішнього завдання робити не можна.",
        "",
        "## Факти для висновку",
        "",
        "Висновок про втрату семантики слід формувати за сукупністю MAE, рангової кореляції та Top-10 overlap, а не лише за explained variance.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=False)

    with PREVIOUS_RESULTS.open("r", encoding="utf-8") as file:
        previous = json.load(file)
    category_words = {
        category: selection["present"]
        for category, selection in previous["words"].items()
    }

    selected = []
    category_words_flat = set()
    for category, words in category_words.items():
        known_words = [word for word in words if word in model.key_to_index]
        selected.extend(known_words)
        category_words_flat.update(known_words)

    candidates = [
        word for word in model.index_to_key
        if suitable_random_word(word, category_words_flat)
    ]
    random.shuffle(candidates)
    random_count = SAMPLE_SIZE - len(selected)
    additional = []
    for word in candidates:
        if word not in category_words_flat and word not in selected:
            additional.append(word)
            if len(additional) == random_count:
                break
    if len(selected) + len(additional) != SAMPLE_SIZE:
        raise RuntimeError("Не вдалося сформувати вибірку потрібного розміру.")

    sample_words = selected + additional
    vectors = np.asarray([model[word] for word in sample_words], dtype=np.float32)

    category_indices = {}
    word_to_index = {word: index for index, word in enumerate(sample_words)}
    for category, words in category_words.items():
        category_indices[category] = np.asarray(
            [word_to_index[word] for word in words if word in word_to_index],
            dtype=np.int64,
        )

    baseline = normalized(vectors)
    rows = []
    pca_reduced_by_dimension = {}
    explained_variance = {}

    for dimensions in PCA_DIMENSIONS:
        pca = PCA(n_components=dimensions, random_state=SEED, svd_solver="randomized")
        reduced = pca.fit_transform(vectors).astype(np.float32)
        pca_reduced_by_dimension[dimensions] = reduced
        explained_variance[dimensions] = float(np.sum(pca.explained_variance_ratio_))

        metrics = evaluate(vectors, reduced)
        rows.append({
            "scope": "overall",
            "method": "PCA",
            "dimensions": dimensions,
            "explained_variance": explained_variance[dimensions],
            "sample_size": SAMPLE_SIZE,
            **metrics,
        })
        for category, indices in category_indices.items():
            metrics = evaluate(vectors[indices], reduced[indices])
            rows.append({
                "scope": category,
                "method": "PCA",
                "dimensions": dimensions,
                "explained_variance": "",
                "sample_size": len(indices),
                **metrics,
            })

    for dimensions in RANDOM_PROJECTION_DIMENSIONS:
        projection = GaussianRandomProjection(n_components=dimensions, random_state=SEED)
        reduced = projection.fit_transform(vectors).astype(np.float32)
        metrics = evaluate(baseline, reduced)
        rows.append({
            "scope": "overall",
            "method": "RandomProjection",
            "dimensions": dimensions,
            "explained_variance": "",
            "sample_size": SAMPLE_SIZE,
            **metrics,
        })
        for category, indices in category_indices.items():
            metrics = evaluate(baseline[indices], reduced[indices])
            rows.append({
                "scope": category,
                "method": "RandomProjection",
                "dimensions": dimensions,
                "explained_variance": "",
                "sample_size": len(indices),
                **metrics,
            })

    write_csv(RESULTS_DIR / "results.csv", rows)

    config = {
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "category_word_count": len(category_words_flat),
        "random_word_count": len(additional),
        "categories": list(category_words.keys()),
        "sample_words": sample_words,
        "model_path": str(MODEL_PATH),
        "pca_dimensions": PCA_DIMENSIONS,
        "random_projection_dimensions": RANDOM_PROJECTION_DIMENSIONS,
        "top_k": TOP_K,
        "selection_criteria": {
            "minimum_word_length": 3,
            "allowed_characters": "Cyrillic letters only",
            "excluded_forms": ["underscores", "hyphens", "spaces", "slashes"],
        },
        "random_projection": "GaussianRandomProjection",
    }
    with (RESULTS_DIR / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    import scipy
    import sklearn

    environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    (RESULTS_DIR / "experiment_report.md").write_text(
        markdown_report(config, rows, environment),
        encoding="utf-8",
    )

    overall_pca = [row for row in rows if row["scope"] == "overall" and row["method"] == "PCA"]
    plt.figure(figsize=(9, 5.5))
    overall_pca.sort(key=lambda row: row["dimensions"])
    plt.plot(
        [row["dimensions"] for row in overall_pca],
        [row["explained_variance"] for row in overall_pca],
        marker="o",
    )
    plt.gca().invert_xaxis()
    plt.xlabel("Кількість вимірів")
    plt.ylabel("Частка збереженої дисперсії")
    plt.title("PCA: explained variance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "explained_variance.png", dpi=180, bbox_inches="tight")
    plt.close()

    save_metric_plot(RESULTS_DIR / "mae.png", rows, "mae", "MAE косинусної подібності", "MAE")
    save_metric_plot(RESULTS_DIR / "spearman.png", rows, "spearman", "Spearman correlation", "ρ Spearman")
    save_metric_plot(RESULTS_DIR / "top10_overlap.png", rows, "top10_overlap", "Збереження Top-10 сусідів", "Середній overlap")

    plt.figure(figsize=(9, 5.5))
    for metric, ylabel in (
        ("mae", "MAE"),
        ("spearman", "ρ Spearman"),
        ("top10_overlap", "Top-10 overlap"),
    ):
        for method in ("PCA", "RandomProjection"):
            method_rows = [
                row for row in rows
                if row["scope"] == "overall"
                and row["method"] == method
                and row["dimensions"] in RANDOM_PROJECTION_DIMENSIONS
            ]
            method_rows.sort(key=lambda row: row["dimensions"])
            if method_rows:
                plt.plot(
                    [row["dimensions"] for row in method_rows],
                    [row[metric] for row in method_rows],
                    marker="o",
                    label=f"{method}: {ylabel}",
                )
    plt.gca().invert_xaxis()
    plt.xlabel("Кількість вимірів")
    plt.title("PCA та Random Projection")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pca_vs_random_projection.png", dpi=180, bbox_inches="tight")
    plt.close()

    print(json.dumps({
        "sample_size": SAMPLE_SIZE,
        "category_word_count": len(category_words_flat),
        "random_word_count": len(additional),
        "rows": len(rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
