
from pathlib import Path

files = [
    "fiction.lowercased.tokenized.word2vec.300d",
    "fiction.lowercased.tokenized.word2vec.300d.bz2",
    "fiction.lowercased.tokenized.300d.bz2"
]

for filename in files:
    path = Path(filename)

    print("\n" + "=" * 60)
    print("Файл:", filename)

    if not path.exists():
        print("Файл не знайдено")
        continue

    print("Розмір:", round(path.stat().st_size / (1024**2), 2), "MB")

    import bz2

    opener = bz2.open if filename.endswith(".bz2") else open

    try:
        with opener(path, "rb") as f:
            data = f.read(200)

        print("Перші байти:", repr(data[:100]))

        try:
            print("Текстовий фрагмент:", data[:150].decode("utf-8"))
        except UnicodeDecodeError:
            print("Формат містить бінарні дані")

    except Exception as e:
        print("Помилка:", e)

from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format(
    "fiction.lowercased.tokenized.word2vec.300d",
    binary=False
)

print("Словник:", len(model))
print("Розмірність:", model.vector_size)
