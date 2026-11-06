import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fiction.lowercased.tokenized.word2vec.300d"
RESULTS_DIR = ROOT / "results" / "experiment_02"
SEED = 20261106
N_PERMUTATIONS = 20_000
MAX_WORDS_PER_CATEGORY = 20

CANDIDATES = OrderedDict([
    ("Фізика", [
        "сила", "енергія", "швидкість", "рух", "прискорення", "маса",
        "сила тяжіння", "тиск", "температура", "тепло", "хвиля", "частинка",
        "атом", "молекула", "електрика", "магнетизм", "заряд", "світло",
        "оптика", "механіка", "кінетика", "потенціал", "рівновага", "імпульс",
        "інерція", "вектор", "кількість", "густина", "фізика"
    ]),
    ("Технології", [
        "комп'ютер", "компютер", "компьютер", "робот", "програма", "процесор",
        "мережа", "алгоритм", "інтернет", "дані", "програмне забезпечення",
        "сервер", "мова", "код", "система", "розробка", "смартфон", "телефон",
        "пристрій", "датчик", "клієнт", "операційна", "мікросхема", "клавіатура",
        "екран", "пам'ять", "штучний інтелект", "автоматизація", "криптографія",
        "обчислення", "інформація", "інформатика", "цифровий", "цифровий", "електроніка",
        "електронний", "програміст", "розробник", "додаток", "інтерфейс", "база даних",
        "технологія", "віртуальний", "прикладний", "сайт", "браузер", "мережевий",
        "плата", "обладнання", "технологічний"
    ]),
    ("Освіта", [
        "учень", "учитель", "школа", "знання", "навчання", "студент",
        "викладач", "університет", "лекція", "урок", "клас", "підручник",
        "домашнє завдання", "іспит", "екзамен", "навчальний", "освіта",
        "педагогіка", "курс", "дисципліна", "семінар", "навичка", "уміння",
        "читання", "письмо", "оцінка", "навчити", "викладання", "заняття",
        "наука"
    ]),
    ("Природа", [
        "вода", "повітря", "сонце", "земля", "рослина", "дерево", "ліс",
        "річка", "озеро", "гора", "море", "океан", "місяць", "небо", "вітер",
        "дощ", "сніг", "камінь", "ґрунт", "квітка", "трава", "тварина",
        "птах", "океан", "природа", "екосистема", "клімат", "зима", "літо",
        "біологія"
    ])
])


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1.0, norms)
    return normalized @ normalized.T


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def category_values(matrix: np.ndarray, spans: dict[str, tuple[int, int]]) -> dict[str, np.ndarray]:
    values = {}
    for category, (start, end) in spans.items():
        values[category] = upper_triangle(matrix[start:end, start:end])
    return values


def label_pair_mean(matrix: np.ndarray, labels: np.ndarray, unique_labels: np.ndarray) -> tuple[float, float]:
    intra = []
    inter = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            value = float(matrix[left, right])
            (intra if labels[left] == labels[right] else inter).append(value)
    return float(np.mean(intra)), float(np.mean(inter))


def write_markdown_report(report: dict) -> None:
    model = report["model"]
    lines = [
        "# Експеримент 02: тематична структурованість семантичного простору",
        "",
        "## Гіпотеза",
        "",
        "Слова однієї тематичної категорії мають у середньому вищу косинусну подібність, ніж слова з різних категорій.",
        "",
        "## Модель",
        "",
        f"- Слів у словнику: {model['word_count']}",
        f"- Розмірність векторів: {model['vector_size']}",
        "",
        "## Відбори категорій",
        "",
    ]
    for category, selection in report["words"].items():
        lines.append(f"### {category}")
        lines.append(f"- Вибрано: {len(selection['present'])}")
        lines.append("- Слова: " + ", ".join(f"`{word}`" for word in selection["present"]))
        lines.append("- Відсутні: " +
                     (", ".join(f"`{word}`" for word in selection["missing"]) if selection["missing"] else "немає"))
        lines.append("")

    lines.extend([
        "## Статистика внутрішьокатегорійної подібності",
        "",
        "| Категорія | Середнє | Медіана | Std | Мінімум | Максимум |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for category, values in report["category_statistics"].items():
        lines.append(
            f"| {category} | {values['mean']:.6f} | {values['median']:.6f} | "
            f"{values['std']:.6f} | {values['min']:.6f} | {values['max']:.6f} |"
        )

    lines.extend([
        "",
        "## Статистика міжкатегорійної подібності",
        "",
        "| Пари категорій | Середнє | Медіана | Std | Мінімум | Максимум |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pair, values in report["cross_category_statistics"].items():
        lines.append(
            f"| {pair} | {values['mean']:.6f} | {values['median']:.6f} | "
            f"{values['std']:.6f} | {values['min']:.6f} | {values['max']:.6f} |"
        )

    test = report["permutation_test"]
    lines.extend([
        "",
        "## Перестановочний тест",
        "",
        f"- Спостережувана різниця внутрішньо — між: {test['observed_difference']:.6f}",
        f"- Перестановок: {test['permutations']}",
        f"- Двостороннє empirical p-value: {test['p_value']:.6f}",
        "",
        "Перестановки виконано на категорійних мітках відібраних слів, а не на окремих попарних спостереженнях.",
        "",
        "## Обмеження",
        "",
        "- Результат залежить від вибраного словника, домену навчальної корпорації та набору термінів.",
        "- Попарні косинусні значення залишаються залежними через спільні вектори.",
        "- Перестановочний тест переставляє мітки зі збереженням розмірів категорій і враховує цю неоднорідність.",
        "- Відсутність деяких термінів може змінювати оцінки; усі відсутні слова наведено в звіті.",
        "- Аналіз описує статичні embeddings і не є прямою оцінкою якості всієї української мови.",
        "",
        "## Обережна інтерпретація",
        "",
        report["interpretation"],
        "",
    ])
    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    words = OrderedDict()
    all_selected = []
    labels = []

    for category, candidates in CANDIDATES.items():
        present = []
        missing = []
        seen = set()
        for word in candidates:
            if word in seen:
                continue
            seen.add(word)
            if word in model.key_to_index:
                if len(present) < MAX_WORDS_PER_CATEGORY:
                    present.append(word)
            else:
                missing.append(word)
        words[category] = {"present": present, "missing": missing, "candidate_count": len(candidates)}
        if len(present) < 15:
            raise RuntimeError(f"{category}: знайдено лише {len(present)} слів; потрібно мінімум 15")
        all_selected.extend(present)
        labels.extend([category] * len(present))

    vectors = np.asarray(model[all_selected], dtype=np.float64)
    similarity = cosine_matrix(vectors)

    spans = {}
    offset = 0
    for category, selection in words.items():
        count = len(selection["present"])
        spans[category] = (offset, offset + count)
        offset += count

    category_stats = {}
    for category, within_values in category_values(similarity, spans).items():
        category_stats[category] = statistics(within_values)

    cross_stats = OrderedDict()
    cross_values = []
    cross_names = list(spans)
    for left_index in range(len(cross_names)):
        for right_index in range(left_index + 1, len(cross_names)):
            left = cross_names[left_index]
            right = cross_names[right_index]
            left_start, left_end = spans[left]
            right_start, right_end = spans[right]
            values = similarity[left_start:left_end, right_start:right_end].reshape(-1)
            cross_values.append(values)
            cross_stats[f"{left}–{right}"] = statistics(values)

    all_cross = np.concatenate(cross_values)
    intra_flat = np.concatenate([values for values in category_values(similarity, spans).values()])
    labels_array = np.asarray(labels)

    observed_intra, observed_cross = label_pair_mean(
        similarity, labels_array, np.unique(labels_array)
    )
    observed_difference = observed_intra - observed_cross

    rng = np.random.default_rng(SEED)
    permutation_differences = np.empty(N_PERMUTATIONS)
    for permutation_index in range(N_PERMUTATIONS):
        shuffled_labels = rng.permutation(labels_array)
        intra_mean, cross_mean = label_pair_mean(
            similarity, shuffled_labels, np.unique(shuffled_labels)
        )
        permutation_differences[permutation_index] = intra_mean - cross_mean

    p_value = (
        np.count_nonzero(np.abs(permutation_differences) >= abs(observed_difference)) + 1
    ) / (N_PERMUTATIONS + 1)

    # Overall similarity matrix and its labels.
    with (RESULTS_DIR / "similarity_matrix.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["слово", *all_selected])
        for word, row in zip(all_selected, similarity):
            writer.writerow([word, *(float(value) for value in row)])

    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(similarity, cmap="viridis", vmin=-1, vmax=1)
    axis.set_xticks(range(len(all_selected)), labels=all_selected, rotation=90, fontsize=7)
    axis.set_yticks(range(len(all_selected)), labels=all_selected, fontsize=7)
    axis.set_title("Матриця косинусної подібності тематичних категорій")
    figure.colorbar(image, ax=axis, label="Косинусна подібність")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "similarity_heatmap.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.boxplot([intra_flat, all_cross], tick_labels=["Внутрішньокатегорійна", "Міжкатегорійна"])
    axis.set_ylabel("Косинусна подібність")
    axis.set_title("Порівняння внутрішньої та міжкатегорійної подібності")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "within_vs_between_boxplot.png", dpi=180)
    plt.close(figure)

    rows = []
    for category, stats in category_stats.items():
        rows.append([category, len(words[category]["present"]), "внутрішня",
                     stats["mean"], stats["median"], stats["std"], stats["min"], stats["max"]])
    for pair, stats in cross_stats.items():
        rows.append([pair, "", "міжкатегорійна",
                     stats["mean"], stats["median"], stats["std"], stats["min"], stats["max"]])
    write_csv(
        RESULTS_DIR / "statistics.csv",
        ["Категорія / пара", "Кількість слів", "Тип", "Середнє", "Медіана", "Std", "Мінімум", "Максимум"],
        rows
    )

    rejected = p_value < 0.05
    if observed_difference > 0 and rejected:
        interpretation = (
            "У цьому наборі спостережувана внутрішньокатегорійна середня подібність перевищує "
            "міжкатегорійну, а двосторонній перестановочний тест дає p < 0.05. Це узгоджується "
            "з гіпотезою для поточного словника та відбору, але не доводить її загальну істинність."
        )
    elif observed_difference > 0:
        interpretation = (
            "У цьому наборі внутрішньокатегорійна середня подібність перевищує міжкатегорійну, "
            "але двосторонній перестановочний тест не показав статистичної значущості на рівні 0.05."
        )
    else:
        interpretation = (
            "У цьому наборі внутрішньокатегорійна середня подібність не перевищує "
            "міжкатегорійну; отримані дані не узгоджуються з формулюванням гіпотези."
        )

    report = {
        "seed": SEED,
        "permutations": N_PERMUTATIONS,
        "model": {
            "path": str(MODEL_PATH),
            "word_count": len(model.key_to_index),
            "vector_size": model.vector_size,
        },
        "words": words,
        "category_statistics": category_stats,
        "cross_category_statistics": cross_stats,
        "overall_between_mean": float(observed_cross),
        "overall_within_mean": float(observed_intra),
        "observed_difference": None,
        "permutation_test": {
            "observed_difference": float(observed_difference),
            "permutations": N_PERMUTATIONS,
            "p_value": float(p_value),
            "alpha": 0.05,
            "significant": bool(rejected),
        },
        "interpretation": interpretation,
    }
    report["observed_difference"] = float(observed_difference)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(report)
    np.savetxt(
        RESULTS_DIR / "permutation_differences.csv",
        permutation_differences,
        delimiter=",",
        header="difference",
        comments=""
    )

    print(json.dumps(report["permutation_test"], ensure_ascii=False, indent=2))
    print(json.dumps({
        "overall_within_mean": report["overall_within_mean"],
        "overall_between_mean": report["overall_between_mean"],
        "observed_difference": report["observed_difference"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
