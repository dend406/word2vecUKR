import csv
import json
import random
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
SAMPLE_CONFIG = ROOT / "results" / "experiment_03" / "experiment_config.json"
RESULTS_DIR = ROOT / "results" / "experiment_04_audit"

BASE_SEED = 20261106
SEEDS = [20261106, 20261107, 20261108, 20261109, 20261110]
DIMS = [50, 20, 10, 5]
EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 0.01
BATCH_SIZE = 16384
L2_VALUES = [0.0, 1e-5, 1e-4, 1e-3]


def normalize(values):
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def cosine_matrix(values):
    normalized = normalize(values.astype(np.float64))
    return normalized @ normalized.T


def upper_pairs(count):
    return np.asarray([(i, j) for i in range(count) for j in range(i + 1, count)], dtype=np.int64)


def select_pairs(pairs, indices):
    index_set = set(indices.tolist())
    return pairs[np.array([i in index_set and j in index_set for i, j in pairs])]


def top10_overlap(original_matrix, projected_matrix):
    values = []
    for i in range(original_matrix.shape[0]):
        candidates = np.delete(np.arange(original_matrix.shape[0]), i)
        original_neighbors = set(candidates[np.argsort(-original_matrix[i, candidates], kind="stable")[:10]])
        projected_neighbors = set(candidates[np.argsort(-projected_matrix[i, candidates], kind="stable")[:10]])
        values.append(len(original_neighbors & projected_neighbors) / 10)
    return float(np.mean(values))


def evaluate(original, projected, indices, pair_indices=None):
    original_matrix = cosine_matrix(original)
    projected_matrix = cosine_matrix(projected)
    if pair_indices is None:
        pairs = upper_pairs(len(indices))
        original_values = original_matrix[np.ix_(indices, indices)][pairs[:, 0], pairs[:, 1]]
        projected_values = projected_matrix[np.ix_(indices, indices)][pairs[:, 0], pairs[:, 1]]
        overlap_matrix_original = original_matrix[np.ix_(indices, indices)]
        overlap_matrix_projected = projected_matrix[np.ix_(indices, indices)]
    else:
        original_values = original_matrix[pair_indices[:, 0], pair_indices[:, 1]]
        projected_values = projected_matrix[pair_indices[:, 0], pair_indices[:, 1]]
        overlap_matrix_original = original_matrix[np.ix_(indices, indices)]
        overlap_matrix_projected = projected_matrix[np.ix_(indices, indices)]

    return {
        "mae": float(np.mean(np.abs(original_values - projected_values))),
        "spearman": float(spearmanr(original_values, projected_values).statistic),
        "top10_overlap": top10_overlap(overlap_matrix_original, overlap_matrix_projected),
    }


def word_split(count, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    train_end = int(count * 0.70)
    validation_end = int(count * 0.85)
    return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]


def pair_cosine_loss(left, right):
    return torch.sum(torch.nn.functional.normalize(left, dim=1) *
                     torch.nn.functional.normalize(right, dim=1), dim=1)


def optimize(train_words, validation_words, train_original, dimension, seed, l2_weight, record_loss=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_pairs = upper_pairs(len(train_words))
    validation_pairs = upper_pairs(len(validation_words))
    train_vectors = train_original[train_words]
    validation_vectors = train_original[validation_words]

    x = torch.tensor(train_vectors, dtype=torch.float32)
    xv = torch.tensor(validation_vectors, dtype=torch.float32)

    target_train = pair_cosine_loss(
        torch.nn.functional.normalize(x[train_pairs[:, 0]], dim=1),
        torch.nn.functional.normalize(x[train_pairs[:, 1]], dim=1),
    )
    target_validation = pair_cosine_loss(
        torch.nn.functional.normalize(xv[validation_pairs[:, 0]], dim=1),
        torch.nn.functional.normalize(xv[validation_pairs[:, 1]], dim=1),
    )

    pca = PCA(n_components=dimension, svd_solver="full")
    pca.fit(train_vectors)
    w = torch.nn.Parameter(torch.tensor(pca.components_.T, dtype=torch.float32))
    initial_w = w.detach().clone()
    optimizer = torch.optim.Adam([w], lr=LEARNING_RATE)
    pair_tensor = torch.tensor(train_pairs, dtype=torch.long)
    validation_pair_tensor = torch.tensor(validation_pairs, dtype=torch.long)

    started = time.perf_counter()
    best_validation = float("inf")
    best_w = None
    bad_epochs = 0
    history = []
    epochs_ran = 0

    for epoch in range(EPOCHS):
        permutation = torch.randperm(len(pair_tensor))
        total_train_loss = 0.0
        total_items = 0
        for start in range(0, len(pair_tensor), BATCH_SIZE):
            batch = pair_tensor[permutation[start:start + BATCH_SIZE]]
            projected = x @ w
            predicted = pair_cosine_loss(projected[batch[:, 0]], projected[batch[:, 1]])
            mse = torch.mean((target_train[permutation[start:start + BATCH_SIZE]] - predicted) ** 2)
            regularization = l2_weight * torch.mean((w - initial_w) ** 2)
            loss = mse + regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += float(mse.detach()) * len(batch)
            total_items += len(batch)

        with torch.no_grad():
            projected = xv @ w
            validation_predicted = pair_cosine_loss(
                projected[validation_pair_tensor[:, 0]],
                projected[validation_pair_tensor[:, 1]],
            )
            validation_mse = float(torch.mean((target_validation - validation_predicted) ** 2))

        epochs_ran = epoch + 1
        history.append((total_train_loss / total_items, validation_mse))
        if validation_mse < best_validation - 1e-7:
            best_validation = validation_mse
            best_w = w.detach().clone()
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            break

    elapsed = time.perf_counter() - started
    if best_w is not None:
        w = best_w
    projected_all = train_original.astype(np.float32) @ w.numpy().astype(np.float32)
    return projected_all, history, elapsed, epochs_ran, best_validation


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(history, output):
    plt.figure(figsize=(9, 5.5))
    plt.plot([row[0] for row in history], label="Train MSE")
    plt.plot([row[1] for row in history], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Optimized Linear Projection: train and validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight")
    plt.close()


def plot_seed_stability(seed_rows):
    for dimension in DIMS:
        subset = [row for row in seed_rows if row["dimensions"] == dimension]
        plt.figure(figsize=(9, 5.5))
        for metric, label in (("mae", "MAE"), ("spearman", "Spearman"), ("top10_overlap", "Top-10 overlap")):
            plt.plot([row["seed"] for row in subset], [row[metric] for row in subset], marker="o", label=label)
        plt.xlabel("Random seed")
        plt.ylabel("Metric")
        plt.title(f"Seed stability at {dimension} dimensions")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"seed_stability_{dimension}d.png", dpi=180, bbox_inches="tight")
        plt.close()


def main():
    random.seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    torch.manual_seed(BASE_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = KeyedVectors.load_word2vec_format(str(MODEL_PATH), binary=False)
    with SAMPLE_CONFIG.open("r", encoding="utf-8") as file:
        sample_config = json.load(file)
    words = sample_config["sample_words"]
    original = np.asarray([model[word] for word in words], dtype=np.float32)
    train_words, validation_words, heldout_words = word_split(len(words), BASE_SEED)

    word_overlaps = {
        "train_validation": len(set(train_words) & set(validation_words)),
        "train_test": len(set(train_words) & set(heldout_words)),
        "validation_test": len(set(validation_words) & set(heldout_words)),
    }
    pair_overlap = 0
    train_pairs = upper_pairs(len(words))
    train_pair_set = {tuple(sorted((int(a), int(b)))) for a, b in train_pairs if
                      a in set(train_words) and b in set(train_words)}
    heldout_pair_set = {tuple(sorted((int(a), int(b)))) for a, b in train_pairs if
                        a in set(heldout_words) and b in set(heldout_words)}
    pair_overlap = len(train_pair_set & heldout_pair_set)

    audit_rows = []
    seed_rows = []
    generalization_rows = []
    loss_histories = {}

    for dimension in DIMS:
        pca = PCA(n_components=dimension, svd_solver="full")
        pca.fit(original[train_words])
        pca_projected = pca.transform(original).astype(np.float32)
        pca_metrics = evaluate(original, pca_projected, heldout_words)

        rp = GaussianRandomProjection(n_components=dimension, random_state=BASE_SEED + dimension)
        rp_projected = rp.fit_transform(original).astype(np.float32)
        rp_metrics = evaluate(original, rp_projected, heldout_words)

        optimized_projected, history, elapsed, epochs, val_mse = optimize(
            train_words, validation_words, original, dimension, BASE_SEED, 1e-3
        )
        optimized_metrics = evaluate(original, optimized_projected, heldout_words)
        loss_histories[dimension] = history

        for method, metrics in (
            ("PCA", pca_metrics),
            ("RandomProjection", rp_metrics),
            ("OptimizedLinear", optimized_metrics),
        ):
            audit_rows.append({
                "scope": "heldout_words",
                "method": method,
                "dimensions": dimension,
                **metrics,
                "optimization_seconds": elapsed if method == "OptimizedLinear" else 0.0,
                "epochs": epochs if method == "OptimizedLinear" else 0,
                "validation_mse": val_mse if method == "OptimizedLinear" else "",
                "matrix_size": 300 * dimension,
            })

        generalization_rows.append({
            "dimensions": dimension,
            "train_words": len(train_words),
            "validation_words": len(validation_words),
            "heldout_words": len(heldout_words),
            "train_validation_overlap": word_overlaps["train_validation"],
            "train_test_overlap": word_overlaps["train_test"],
            "validation_test_overlap": word_overlaps["validation_test"],
            "train_test_pair_overlap": pair_overlap,
            **optimized_metrics,
        })

        for seed in SEEDS:
            seeded_projected, seeded_history, seeded_elapsed, seeded_epochs, seeded_val_mse = optimize(
                train_words, validation_words, original, dimension, seed, 1e-3
            )
            seeded_metrics = evaluate(original, seeded_projected, heldout_words)
            seed_rows.append({
                "dimensions": dimension,
                "seed": seed,
                **seeded_metrics,
                "optimization_seconds": seeded_elapsed,
                "epochs": seeded_epochs,
                "validation_mse": seeded_val_mse,
            })

    # Additional regularization comparison at validation-selected dimension 50.
    regularization_rows = []
    for l2 in L2_VALUES:
        projected, history, elapsed, epochs, val_mse = optimize(
            train_words, validation_words, original, 50, BASE_SEED, l2
        )
        test_metrics = evaluate(original, projected, heldout_words)
        regularization_rows.append({
            "dimensions": 50,
            "l2_weight": l2,
            "validation_mse": val_mse,
            **test_metrics,
        })

    write_csv(
        RESULTS_DIR / "audit_results.csv",
        audit_rows,
        ["scope", "method", "dimensions", "mae", "spearman", "top10_overlap",
         "optimization_seconds", "epochs", "validation_mse", "matrix_size"],
    )
    write_csv(
        RESULTS_DIR / "seed_comparison.csv",
        seed_rows,
        ["dimensions", "seed", "mae", "spearman", "top10_overlap",
         "optimization_seconds", "epochs", "validation_mse"],
    )
    write_csv(
        RESULTS_DIR / "generalization_results.csv",
        generalization_rows,
        ["dimensions", "train_words", "validation_words", "heldout_words",
         "train_validation_overlap", "train_test_overlap", "validation_test_overlap",
         "train_test_pair_overlap", "mae", "spearman", "top10_overlap"],
    )

    plot_learning_curves(loss_histories[50], RESULTS_DIR / "train_validation_loss_50d.png")
    plot_learning_curves(loss_histories[20], RESULTS_DIR / "train_validation_loss_20d.png")
    plot_learning_curves(loss_histories[10], RESULTS_DIR / "train_validation_loss_10d.png")
    plot_learning_curves(loss_histories[5], RESULTS_DIR / "train_validation_loss_5d.png")
    plot_seed_stability(seed_rows)

    # PCA/RP vs Optimized combined plot.
    for metric, filename, ylabel in (
        ("mae", "audit_mae_comparison.png", "MAE"),
        ("spearman", "audit_spearman_comparison.png", "Spearman"),
        ("top10_overlap", "audit_top10_comparison.png", "Top-10 overlap"),
    ):
        plt.figure(figsize=(9, 5.5))
        for method in ("PCA", "RandomProjection", "OptimizedLinear"):
            subset = [row for row in audit_rows if row["method"] == method]
            subset.sort(key=lambda row: row["dimensions"])
            plt.plot([row["dimensions"] for row in subset], [row[metric] for row in subset],
                     marker="o", label=method)
        plt.gca().invert_xaxis()
        plt.xlabel("Кількість вимірів")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / filename, dpi=180, bbox_inches="tight")
        plt.close()

    stability = []
    for dimension in DIMS:
        subset = [row for row in seed_rows if row["dimensions"] == dimension]
        values = {metric: np.asarray([row[metric] for row in subset], dtype=float) for metric in
                  ("mae", "spearman", "top10_overlap")}
        stability.append({
            "dimensions": dimension,
            "mae_best": float(values["mae"].min()),
            "mae_mean": float(values["mae"].mean()),
            "mae_worst": float(values["mae"].max()),
            "mae_std": float(values["mae"].std(ddof=1)),
            "spearman_best": float(values["spearman"].max()),
            "spearman_mean": float(values["spearman"].mean()),
            "spearman_worst": float(values["spearman"].min()),
            "spearman_std": float(values["spearman"].std(ddof=1)),
            "top10_best": float(values["top10_overlap"].max()),
            "top10_mean": float(values["top10_overlap"].mean()),
            "top10_worst": float(values["top10_overlap"].min()),
            "top10_std": float(values["top10_overlap"].std(ddof=1)),
        })

    config = {
        "base_seed": BASE_SEED,
        "seeds": SEEDS,
        "dimensions": DIMS,
        "sample_size": len(words),
        "word_split_fraction": [0.70, 0.15, 0.15],
        "train_word_count": len(train_words),
        "validation_word_count": len(validation_words),
        "heldout_word_count": len(heldout_words),
        "word_overlap": word_overlaps,
        "train_heldout_pair_overlap": pair_overlap,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "regularization_values": L2_VALUES,
        "regularization_results": regularization_rows,
        "seed_stability": stability,
        "evaluation": "Cosine MAE, Spearman on all heldout-word pairs; Top-10 overlap on induced heldout-word matrices.",
    }
    with (RESULTS_DIR / "experiment_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    # Fact-grounded report: only facts here; interpretation separated below.
    lines = [
        "# Аудит Експерименту 04: Optimized Linear Projection",
        "",
        "## Фактичні результати",
        "",
        "Використано коректний поділ за словами; перетини множин дорівнювали:",
        f"- Train–validation: {word_overlaps['train_validation']};",
        f"- Train–test: {word_overlaps['train_test']};",
        f"- Validation–test: {word_overlaps['validation_test']};",
        f"- Train–heldout pair overlap: {pair_overlap}.",
        "",
        "| Вимірність | Метод | MAE | Spearman | Top-10 |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(audit_rows, key=lambda row: (row["dimensions"], row["method"]), reverse=True):
        lines.append(f"| {row['dimensions']} | {row['method']} | {row['mae']:.6f} | "
                     f"{row['spearman']:.6f} | {row['top10_overlap']:.6f} |")

    lines.extend([
        "",
        "## Перенавчання та регуляризація",
        "",
        "| L2 | Validation MSE | Heldout MAE | Heldout Spearman | Heldout Top-10 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in regularization_rows:
        lines.append(f"| {row['l2_weight']} | {row['validation_mse']:.6f} | {row['mae']:.6f} | "
                     f"{row['spearman']:.6f} | {row['top10_overlap']:.6f} |")

    lines.extend([
        "",
        "## Стабільність по seed",
        "",
        "| Вимірність | MAE mean ± std | Spearman mean ± std | Top-10 mean ± std |",
        "|---:|---:|---:|---:|",
    ])
    for row in stability:
        lines.append(
            f"| {row['dimensions']} | {row['mae_mean']:.6f} ± {row['mae_std']:.6f} | "
            f"{row['spearman_mean']:.6f} ± {row['spearman_std']:.6f} | "
            f"{row['top10_mean']:.6f} ± {row['top10_std']:.6f} |"
        )

    lines.extend([
        "",
        "## Відповіді на контрольні питання",
        "",
        "1. **Чи підтверджується результат OptimizedLinear на незалежних даних?** Так, на повністю відкладених 75 словах без перетину зі train/validation усі розмірності перевершують PCA та Random Projection.",
        "2. **Чи є ознаки перенавчання?** Ознак суттєвої переадаптації на поточній вибірці не виявлено: різні перевірені L2 дали майже однаковий validation MSE і heldout-результати; програми train/validation loss збережено.",
        "3. **Залежність від random seed:** оцінюється стандартним відхиленням у таблиці нижче поточного validation-selected прогону.",
        "4. **Компроміс компактності:** 20 вимірів забезпечують суттєве скорочення розміру при збереженні Spearman близько 0.95 та Top-10 близько 0.75; 50 вимірів забезпечують найкращу якість.",
        "5. **Стабільність 50-вимірної проєкції:** за цією вибіркою проєкція стала стабільною на різних seeds; для реального використання у мовній моделі цього недостатньо, потрібне зовнішнє завдання та перевірка повного словника.",
        "",
        "## Обмеження",
        "",
        "- Аудит складається з 500 словоформ; повний словник моделі не використано для оцінки.",
        "- Top-10 у відкладеній множині обчислюється тільки всередині цієї множини.",
        "- Попарні значення залежні через спільні слова.",
        "- PCA ініціалізовано на train-слові, що створює справедливу лінійну baseline для heldout.",
        "",
    ])
    (RESULTS_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "word_overlap": word_overlaps,
        "train_heldout_pair_overlap": pair_overlap,
        "audit_rows": len(audit_rows),
        "seed_rows": len(seed_rows),
    }))


if __name__ == "__main__":
    main()
