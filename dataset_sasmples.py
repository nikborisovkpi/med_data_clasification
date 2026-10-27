"""
OrganAMNIST (size=224): завантаження В ПАПКУ ПРОЄКТУ, відбір класів і перегляд
зображень БЕЗ завантаження всього датасету в оперативну пам'ять.

Вибрані класи: серце, нирки (ліва й права), печінка, легені (ліва й права).

Як це економить пам'ять:
  * Файл .npz (архів) завантажується на диск у <папка_проєкту>/data/medmnist/.
  * З архіву на диск по одному масиву (потоково, шматками) розпаковуються .npy-файли.
  * .npy відкриваються через np.memmap: зображення лежать на диску, а в RAM
    підтягуються лише ті, до яких ви звертаєтесь.
  * Відбір класів - це лише масив індексів (кілька МБ), копії зображень не робляться.

Місце на диску: .npz + приблизно 3 ГБ розпакованих .npy
(можна видалити .npz після розпакування: DELETE_NPZ_AFTER_EXTRACT = True).

Залежності:
    pip install medmnist numpy matplotlib
"""

import shutil
import zipfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO

# ---------------------------------------------------------------- налаштування
FLAG = "organamnist"
IMG_SIZE = 224
SPLITS = ["train", "val", "test"]

SELECTED_CLASSES = [
    "heart",
    "kidney-left",
    "kidney-right",
    "liver",
    "lung-left",
    "lung-right",
    "spleen",
    "pancreas",
]

PROJECT_DIR = Path(__file__).resolve().parent          # папка, де лежить скрипт
DATA_DIR = PROJECT_DIR / "data" / "medmnist"           # сюди вантажиться датасет
NPZ_PATH = DATA_DIR / f"{FLAG}_{IMG_SIZE}.npz"
NPY_DIR = DATA_DIR / f"{FLAG}_{IMG_SIZE}"              # сюди розпаковуються .npy
DELETE_NPZ_AFTER_EXTRACT = False

SEED = 42
NPY_KEYS = [f"{s}_{k}" for s in SPLITS for k in ("images", "labels")]


# ------------------------------------------------- завантаження та розпакування
def extract_npz(npz_path: Path, out_dir: Path, keys):
    """Потокова розпаковка масивів з .npz у окремі .npy (без завантаження в RAM)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as zf:
        members = set(zf.namelist())
        for key in keys:
            member = f"{key}.npy"
            target = out_dir / member
            if target.exists():
                continue
            if member not in members:
                raise KeyError(f"У {npz_path.name} немає масиву {member}")
            print(f"Розпаковую {member} ...")
            tmp = target.with_suffix(".npy.part")
            with zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
            tmp.replace(target)          # файл з'являється лише після повного запису


def ensure_dataset():
    """Завантажує й розпаковує датасет у папку проєкту (якщо ще не зроблено)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if all((NPY_DIR / f"{k}.npy").exists() for k in NPY_KEYS):
        print(f"Датасет уже розпакований: {NPY_DIR}")
        return

    # Той самий механізм завантаження, що й у medmnist (з перевіркою MD5).
    # Якщо файл уже є й цілий, повторно він не завантажується.
    from torchvision.datasets.utils import download_url

    download_url(
        url=INFO[FLAG][f"url_{IMG_SIZE}"],
        root=str(DATA_DIR),
        filename=NPZ_PATH.name,
        md5=INFO[FLAG][f"MD5_{IMG_SIZE}"],
    )
    extract_npz(NPZ_PATH, NPY_DIR, NPY_KEYS)

    if DELETE_NPZ_AFTER_EXTRACT:
        NPZ_PATH.unlink()
        print(f"Видалено {NPZ_PATH.name}")


# ------------------------------------------------------------------ дані
class OrganSubset:
    """
    Підмножина вибраних класів одного спліту. Зображення НЕ копіюються:
    зберігається memmap на весь спліт і масив індексів потрібних зображень.
    Мітки перенумеровані в 0..K-1 у порядку class_names.
    """

    def __init__(self, split, class_names):
        name_to_id = {name: int(i) for i, name in INFO[FLAG]["label"].items()}
        missing = [c for c in class_names if c not in name_to_id]
        if missing:
            raise ValueError(f"Невідомі класи: {missing}")

        images = np.load(NPY_DIR / f"{split}_images.npy", mmap_mode="r")
        if images.ndim == 4:                             # (N,H,W,1) -> (N,H,W)
            images = images[..., 0]
        all_labels = np.load(NPY_DIR / f"{split}_labels.npy").reshape(-1)

        old_ids = [name_to_id[c] for c in class_names]
        lut = np.full(len(name_to_id), -1, dtype=np.int64)
        for new_id, old_id in enumerate(old_ids):
            lut[old_id] = new_id

        self.split = split
        self.class_names = list(class_names)
        self.images = images                              # np.memmap (на диску)
        self.indices = np.flatnonzero(np.isin(all_labels, old_ids))
        self.labels = lut[all_labels[self.indices]]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        """Повертає (зображення HxW uint8 - копія в RAM, мітка 0..K-1)."""
        img = np.array(self.images[self.indices[i]])
        return img, int(self.labels[i])


def print_class_counts(subsets, class_names):
    header = f"{'клас':<15}" + "".join(f"{s:>10}" for s in subsets)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<15}"
        for sub in subsets.values():
            row += f"{int((sub.labels == i).sum()):>10}"
        print(row)
    print("-" * len(header))
    print(f"{'усього':<15}" + "".join(f"{len(s):>10}" for s in subsets.values()))


def show_samples(subset, n_per_class=6, seed=0, title="", save_path=None):
    """Показує приклади зображень для кожного класу."""
    rng = np.random.default_rng(seed)
    class_names = subset.class_names
    n_classes = len(class_names)
    classes_per_row = 2
    class_rows = (n_classes + classes_per_row - 1) // classes_per_row

    fig, axes = plt.subplots(
        class_rows, classes_per_row * n_per_class,
        figsize=(1.9 * classes_per_row * n_per_class, 2.3 * class_rows),
        squeeze=False,
    )
    for class_id, name in enumerate(class_names):
        class_row = class_id // classes_per_row
        class_col = class_id % classes_per_row
        first_col = class_col * n_per_class
        positions = np.flatnonzero(subset.labels == class_id)
        chosen = rng.choice(positions, size=min(n_per_class, len(positions)),
                            replace=False)
        for col in range(n_per_class):
            ax = axes[class_row, first_col + col]
            if col < len(chosen):
                img, _ = subset[chosen[col]]              # читається лише 1 зображення
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[class_row, first_col].set_title(name, fontsize=11)

    # Приховуємо незайняті групи в останньому рядку.
    for class_id in range(n_classes, class_rows * classes_per_row):
        class_row = class_id // classes_per_row
        first_col = (class_id % classes_per_row) * n_per_class
        for col in range(n_per_class):
            axes[class_row, first_col + col].set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Збережено: {save_path}")
    plt.show()


# ---------------------------------------------------------------------- main
def main():
    ensure_dataset()

    subsets = {s: OrganSubset(s, SELECTED_CLASSES) for s in SPLITS}
    for s, sub in subsets.items():
        print(f"[{s}] memmap: {isinstance(sub.images, np.memmap)}, "
              f"усього у файлі: {sub.images.shape}, вибрано: {len(sub)}")

    print()
    print_class_counts(subsets, SELECTED_CLASSES)

    show_samples(
        subsets["train"],
        n_per_class=3,
        seed=SEED,
        title=f"OrganAMNIST {IMG_SIZE}x{IMG_SIZE} - приклади по класах (train)",
        save_path=PROJECT_DIR / "organamnist_samples.png",
    )


if __name__ == "__main__":
    main()