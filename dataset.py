"""
Задача 1: Формування датасету на базі OrganAMNIST (224x224).

Класи:
  0: liver  (печінка)                     <- ["liver"]
  1: kidney (нирки: ліва + права)         <- ["kidney-left", "kidney-right"]
  2: lung   (легені: ліва + права)        <- ["lung-left", "lung-right"]
  3: other  (інше: серце, селезінка,      <- ["heart", "spleen", "pancreas"]
             підшлункова залоза)

Клас "other" дозволяє моделі розпізнавати нецільові органи та не видавати
помилкових позитивних рішень (false positives) для основних онкопайплайнів.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
try:
    from torch.utils.data import Dataset
except ImportError:
    Dataset = object
from medmnist import INFO

from dataset_sasmples import (
    FLAG,
    IMG_SIZE,
    SPLITS,
    PROJECT_DIR,
    NPY_DIR,
    ensure_dataset,
)

# Групування підкласів OrganAMNIST: 3 основні цільові органи + клас "other"
TARGET_ORGANS = {
    "liver": ["liver"],
    "kidney": ["kidney-left", "kidney-right"],
    "lung": ["lung-left", "lung-right"],
    "other": ["heart", "spleen", "pancreas"],
}
CLASS_NAMES = list(TARGET_ORGANS.keys())


class OrganClassificationDataset(Dataset):
    """
    PyTorch-сумісний датасет для класифікації органів,
    що читає зображення з диску через np.memmap.
    """

    def __init__(self, split: str, organ_groups: dict = TARGET_ORGANS, transform=None):
        if split not in SPLITS:
            raise ValueError(f"Невідомий спліт '{split}'. Доступні: {SPLITS}")

        name_to_raw_id = {name: int(i) for i, name in INFO[FLAG]["label"].items()}

        images = np.load(NPY_DIR / f"{split}_images.npy", mmap_mode="r")
        if images.ndim == 4:
            images = images[..., 0]
        all_labels = np.load(NPY_DIR / f"{split}_labels.npy").reshape(-1)

        # Таблиця перекодування сирих ID OrganAMNIST -> 0..3 (або -1 для відкинутих класів)
        lut = np.full(len(name_to_raw_id), -1, dtype=np.int64)
        for new_id, (_, raw_names) in enumerate(organ_groups.items()):
            for raw_name in raw_names:
                if raw_name not in name_to_raw_id:
                    raise ValueError(f"Невідомий клас у OrganAMNIST: {raw_name}")
                lut[name_to_raw_id[raw_name]] = new_id

        valid_mask = lut[all_labels] != -1
        self.split = split
        self.class_names = list(organ_groups.keys())
        self.images = images
        self.indices = np.flatnonzero(valid_mask)
        self.labels = lut[all_labels[self.indices]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        raw_img = np.array(self.images[self.indices[idx]], dtype=np.uint8)
        label = int(self.labels[idx])

        if self.transform is not None:
            pil_img = Image.fromarray(raw_img, mode="L").convert("RGB")
            img_tensor = self.transform(pil_img)
            return img_tensor, label

        return raw_img, label



def print_dataset_summary(datasets: dict):
    header = f"{'Клас (орган)':<18}" + "".join(f"{s:>10}" for s in datasets)
    print(header)
    print("-" * len(header))
    for class_id, name in enumerate(CLASS_NAMES):
        row = f"{name:<18}"
        for ds in datasets.values():
            count = int((ds.labels == class_id).sum())
            row += f"{count:>10}"
        print(row)
    print("-" * len(header))
    print(f"{'Усього':<18}" + "".join(f"{len(ds):>10}" for ds in datasets.values()))


def save_dataset_samples(dataset: OrganClassificationDataset, n_per_class: int = 6, seed: int = 42, save_path: Path = None):
    rng = np.random.default_rng(seed)
    n_classes = len(dataset.class_names)

    fig, axes = plt.subplots(
        n_classes,
        n_per_class,
        figsize=(2.0 * n_per_class, 2.0 * n_classes),
        squeeze=False,
    )

    for row, name in enumerate(dataset.class_names):
        positions = np.flatnonzero(dataset.labels == row)
        chosen = rng.choice(positions, size=min(n_per_class, len(positions)), replace=False)
        for col in range(n_per_class):
            ax = axes[row, col]
            if col < len(chosen):
                img, _ = dataset[chosen[col]]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[row, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=12, fontweight="bold", labelpad=18)

    fig.suptitle(f"Датасет органів ({IMG_SIZE}x{IMG_SIZE}): 3 основні органи + 'other' (heart/spleen/pancreas)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Збережено приклади класів: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    ensure_dataset()
    datasets = {s: OrganClassificationDataset(s) for s in SPLITS}
    print("\nСтатистика оновленого датасету (3 основні класи + other):")
    print_dataset_summary(datasets)
    save_dataset_samples(
        datasets["train"],
        save_path=PROJECT_DIR / "dataset_classes_samples.png",
    )
