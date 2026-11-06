import csv
import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fiction.lowercased.tokenized.word2vec.300d"
EXPERIMENT_03_CONFIG = ROOT / "results" / "experiment_03" / "experiment_config.json"
RESULTS_DIR = ROOT / "results" / "experiment_04"

SEED = 20261106
DIMS = [50, 20, 10, 5]
TOP_K = 10
EPOCHS = 300
PATIENCE = 25
LEARNING_RATE = 0.01
L2_WEIGHT = 1e-4
BATCH_SIZE = 16384


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0.0, 1.0, norms)


def cosine_matrix(values: np.ndarray) -> np.ndarray:
    values = normalize_rows(values.astype(np.float64))
    return values @ values.T


def pair_values(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return matrix[pairs[:, 0], pairs[:, 1]]


def top10_overlap(original_matrix: np.ndarray, projected_matrix: np.ndarray) -> float:
    values = []
    count = min(TOP_K, original_matrix.shape[0] - 1)
    for index in range(original_matrix.shape[0]):
        candidates = np.delete(np.arange(original_matrix.shape[0]), index)
        original_order = np.argsort(-original_matrix[index, candidates], kind="stable")[:count]
        projected_order = np.argsort(-projected_matrix[index, candidates], kind="stable")[:count]
        values.append(
            len(set(candidates[original_order]) & set(candidates[projected_order])) / count
        )
    return float(np.mean(values))


def evaluate(original: np.ndarray, projected: np.ndarray, pairs: np.ndarray) -> dict[str, float]:
    original_cosine = cosine_matrix(original)
    projected_cosine = cosine_matrix(projected)
    expected = pair_values(original_cosine, pairs)
    actual = pair_values(projected_cosine, pairs)
    correlation = float(spearmanr(expected, actual).statistic)
    if not np.isfinite(correlation):
        correlation = 0.0
    return {
        "mae": float(np.mean(np.abs(expected - actual))),
        "spearman": correlation,
        "top10_overlap": top10_overlap(original_cosine, projected_cosine),
    }


def split_pairs(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = np.asarray(
        [(left, right) for left in range(count) for right in range(left + 1, count)],
        dtype=np.int64,
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    train_end = int(len(pairs) * 0.70)
    validation_end = int(len(pairs) * 0.85)
    return pairs[:train_end], pairs[train_end:validation_end], pairs[validation_end:]


def normalized_cosine_loss(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = torch.nn.functional.normalize(left, p=2, dim=1)
    right = torch.nn.functional.normalize(right, p=2, dim=1)
    return torch.sum(left * right, dim=1)


def optimize_projection(
    original: np.ndarray,
    pairs_train: np.ndarray,
    pairs_validation: np.ndarray,
    dimension: int,
) -> tuple[np.ndarray, float, int, float]:
    torch.manual_seed(SEED + dimension)
    np.random.seed(SEED + dimension)

    original_torch = torch.as_tensor(original, dtype=torch.float32)
    original_normalized = torch.nn.functional.normalize(original_torch, p=2, dim=1)
    train_torch = torch.as_tensor(pairs_train, dtype=torch.long)
    validation_torch = torch.as_tensor(pairs_validation, dtype=torch.long)
    target_train = normalized_cosine_loss(
        original_normalized[train_torch[:, 0]],
        original_normalized[train_torch[:, 1]],
    )
    target_validation = normalized_cosine_loss(
        original_normalized[validation_torch[:, 0]],
        original_normalized[validation_torch[:, 1]],
    )

    pca = PCA(n_components=dimension, svd_solver="full")
    pca.fit(original)
    initial = pca.components_.T.astype(np.float32)
    # PCA.transform uses centering; optimization itself remains a linear X @ W map.
    projection = torch.nn.Parameter(torch.from_numpy(initial.copy()))
    initial_torch = projection.detach().clone()
    optimizer = torch.optim.Adam([projection], lr=LEARNING_RATE)

    started = time.perf_counter()
    best_validation = float("inf")
    best_state = None
    patience_counter = 0
    epochs_ran = 0

    for epoch in range(EPOCHS):
        permutation = torch.randperm(len(train_torch))
        for start in range(0, len(train_torch), BATCH_SIZE):
            batch_indices = permutation[start:start + BATCH_SIZE]
            batch = train_torch[batch_indices]
            projected = original_torch @ projection
            projected_cosine = normalized_cosine_loss(
                projected[batch[:, 0]],
                projected[batch[:, 1]],
            )
            mse = torch.mean((target_train[batch_indices] - projected_cosine) ** 2)
            regularization = L2_WEIGHT * torch.mean((projection - initial_torch) ** 2)
            loss = mse + regularization

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            projected = original_torch @ projection
            validation_cosine = normalized_cosine_loss(
                projected[validation_torch[:, 0]],
                projected[validation_torch[:, 1]],
            )
            validation_mse = torch.mean((target_validation - validation_cosine) ** 2).item()

        epochs_ran = epoch + 1
        if validation_mse < best_validation - 1e-7:
            best_validation = validation_mse
            best_state = projection.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    elapsed = time.perf_counter() - started
    if best_state is not None:
        projection = best_state
    result = (original.astype(np.float32) @ projection.numpy()).astype(np.float32)
    return result, elapsed, epochs_ran, best_validation


def save_plot(path: Path, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    methods = ["PCA", "RandomProjection", "OptimizedLinear"]
    plt.figure(figsize=(9, 5.5))
    for method in methods:
        selected = [row for row in rows if row["scope"] == "overall" and row["method"] == method]
        selected.sort(key=lambda row: row["dimensions"])
        if selected:
            plt.plot(
                [row["dimensions"] for row in selected],
                [row[metric] for row in selected],
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


def write_results(rows: list[dict]) -> None:
    fieldnames = [
        "scope", "method", "dimensions", "mae", "spearman", "top10_overlap",
        "optimization_seconds", "epochs", "validation_mse", "matrix_size",
    ]
    with (RESULTS_DIR / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict], config: dict) -> None:
    lines = [
        "# Експеримент 04: оптимізована лінійна проєкція Word2Vec",
        "",
        "## Перевірка експерименту 03",
        "",
        "- PCA sklearn центрує навчальні дані перед знаходженням компонент.",
        "- Косинусна подібність обчислюється після нормалізації вихідних і зменшених векторів.",
        "- MAE обчислюється як середнє абсолютної різниці між однаковими верхніми трікутниками пар косинусів.",
        "- У попередньому коді PCA, Random Projection та категорійні оцінки використовують той самий набір випадків, але попередня загальна оцінка охоплює повний набір у верхньому трікутнику.",
        "- `MAE(50) < MAE(100)` можливий, бо PCA мінімізує реконструкційний MSE центрованих даних, а не MAE косинусної подібності; крім того, косинуси залежать від нормалізації кожного вектора. Отже, різниця дисперсії не є монотонною функцією MAE подібності, а randomized SVD також може давати невеликі числові/алгоритмічні відмінності.",
        "- У цьому експерименті всі порівнювані методи оцінюються на однаковому фіксованому наборі тестових пар.",
        "",
        "## Конфігурація",
        "",
        f"- Seed: `{config['seed']}`",
        f"- Training pairs: `{config['train_pair_count']}`",
        f"- Validation pairs: `{config['validation_pair_count']}`",
        f"- Test pairs: `{config['test_pair_count']}`",
        f"- Dimensions: `{config['dimensions']}`",
        f"- Epochs: `{config['epochs']}`",
        f"- Learning rate: `{config['learning_rate']}`",
        f"- L2 regularization: `{config['l2_weight']}`",
        "",
        "## Результати на загальних тестових парах",
        "",
        "| Вимірність | Метод | MAE | Spearman | Top-10 | Час, с | Епохи | Матриця |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        (item for item in rows if item["scope"] == "overall"),
        key=lambda item: (-item["dimensions"], item["method"]),
    ):
        lines.append(
            f"| {row['dimensions']} | {row['method']} | {row['mae']:.6f} | "
            f"{row['spearman']:.6f} | {row['top10_overlap']:.6f} | "
            f"{row['optimization_seconds']:.3f} | {row['epochs']} | {row['matrix_size']} |"
        )

    lines.extend([
        "",
        "## Обмеження",
        "",
        "- Оптимізація використовує часткову вибірку з 500 словоформ і обмежений набір пар.",
        "- Цільова функція оптимізує парну косинусну подібність, але це не гарантує покращення задачі нижчого рівня.",
        "- Train/validation/test split обмежує витік пар; однакові слова можуть бути присутні в декількох парах, тому спостереження залишаються залежними.",
        "- Лінійна проєкція та PCA є окремими моделями представлення; регуляризація спрямована до PCA-ініціалізації.",
        "- Результати залежать від seed, навчальної корпорації та обраного словника.",
        "",
    ])
    (RESULTS_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=False)
    with EXPERIMENT_03_CONFIG.open("r", encoding="utf-8") as file:
        config_03 = json.load(file)
    words = config_03["sample_words"]
    original = np.asarray([model[word] for word in words], dtype=np.float32)

    train_pairs, validation_pairs, test_pairs = split_pairs(len(words), SEED)
    rows = []
    optimization_info = {}

    for dimension in DIMS:
        pca = PCA(n_components=dimension, svd_solver="full")
        pca_projected = pca.fit_transform(original).astype(np.float32)
        metrics = evaluate(original, pca_projected, test_pairs)
        rows.append({
            "scope": "overall",
            "method": "PCA",
            "dimensions": dimension,
            **metrics,
            "optimization_seconds": 0.0,
            "epochs": 0,
            "validation_mse": "",
            "matrix_size": original.shape[1] * dimension,
        })

        projection = GaussianRandomProjection(
            n_components=dimension,
            random_state=SEED + dimension,
        )
        random_projected = projection.fit_transform(original).astype(np.float32)
        metrics = evaluate(original, random_projected, test_pairs)
        rows.append({
            "scope": "overall",
            "method": "RandomProjection",
            "dimensions": dimension,
            **metrics,
            "optimization_seconds": 0.0,
            "epochs": 0,
            "validation_mse": "",
            "matrix_size": original.shape[1] * dimension,
        })

        optimized_projected, elapsed, epochs_ran, validation_mse = optimize_projection(
            original,
            train_pairs,
            validation_pairs,
            dimension,
        )
        metrics = evaluate(original, optimized_projected, test_pairs)
        rows.append({
            "scope": "overall",
            "method": "OptimizedLinear",
            "dimensions": dimension,
            **metrics,
            "optimization_seconds": elapsed,
            "epochs": epochs_ran,
            "validation_mse": validation_mse,
            "matrix_size": original.shape[1] * dimension,
        })
        optimization_info[str(dimension)] = {
            "optimization_seconds": elapsed,
            "epochs": epochs_ran,
            "validation_mse": validation_mse,
        }

    write_results(rows)

    config = {
        "seed": SEED,
        "dimensions": DIMS,
        "sample_size": len(words),
        "train_pair_fraction": 0.70,
        "validation_pair_fraction": 0.15,
        "test_pair_fraction": 0.15,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "test_pair_count": len(test_pairs),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "l2_weight": L2_WEIGHT,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "loss": "mean((cos_original - cos_projected)^2)",
        "pair_split": "deterministic shuffle of all unique word pairs",
        "optimization_info": optimization_info,
    }
    with (RESULTS_DIR / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
    write_report(rows, config)

    save_plot(RESULTS_DIR / "mae_comparison.png", rows, "mae", "MAE косинусної подібності", "MAE")
    save_plot(RESULTS_DIR / "spearman_comparison.png", rows, "spearman", "Збереження рангової структури", "ρ Spearman")
    save_plot(RESULTS_DIR / "top10_comparison.png", rows, "top10_overlap", "Top-10 overlap", "Середній overlap")

    print(json.dumps({
        "rows": len(rows),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "test_pairs": len(test_pairs),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
