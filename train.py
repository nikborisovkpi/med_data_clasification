"""
Задача 2: Навчання та оцінка якості моделі класифікації органів (liver, kidney, lung, other).

Модель: ResNet-18 (Transfer Learning з вагами ImageNet + адаптація під 4 класи:
3 основні органи + клас "other" для відхилення сторонніх органів heart/spleen/pancreas).
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from dataset_sasmples import PROJECT_DIR, ensure_dataset
from dataset import OrganClassificationDataset, CLASS_NAMES

# ---------------------------------------------------------------- налаштування
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "organ_resnet18.pth"
METRICS_PATH = MODEL_DIR / "metrics.json"
CM_PLOT_PATH = PROJECT_DIR / "confusion_matrix.png"
LOSS_PLOT_PATH = PROJECT_DIR / "training_curves.png"

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0  # безпечно для Windows та np.memmap
SEED = 42


def get_transforms():
    """Трансформації для навчання (з аугментацією) та валідації/тесту."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """Створює ResNet-18 із заміненим класифікаційним шаром."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Обчислює ваги класів для компенсації дисбалансу (оскільки kidney та lung об'єднують лівий+правий орган)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets, all_probs = [], [], []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred, y_prob


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix (Test Split - 3 класи органів)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Збережено Confusion Matrix: {save_path}")


def save_training_curves(history: dict, save_path: Path):
    """Будує та зберігає графіки динаміки Loss та Accuracy для Train і Validation."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))

    # Графік Loss
    ax_loss.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax_loss.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax_loss.set_title("Динаміка функції втрат (Loss)", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Епоха", fontsize=11)
    ax_loss.set_ylabel("Loss", fontsize=11)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(fontsize=11)

    # Графік Accuracy
    ax_acc.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    ax_acc.plot(epochs, [a * 100 for a in history["val_acc"]], "s--", label="Val Accuracy", color="#d62728", linewidth=2)
    ax_acc.set_title("Динаміка точності (Accuracy %)", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Епоха", fontsize=11)
    ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Збережено графіки навчання: {save_path}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
 
    ensure_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій для навчання: {device}")
 
    train_tf, eval_tf = get_transforms()
    train_ds = OrganClassificationDataset("train", transform=train_tf)
    val_ds = OrganClassificationDataset("val", transform=eval_tf)
    test_ds = OrganClassificationDataset("test", transform=eval_tf)
 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
 
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=True).to(device)
 
    class_weights = compute_class_weights(train_ds.labels, len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
 
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
 
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
 
        progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch:>2}/{EPOCHS}",
            unit="step",
            ascii=" ╸━",  # пусте / «курсор»
            colour="cyan",
            dynamic_ncols=True,
            bar_format=(
                "{desc} {percentage:3.0f}% ┃{bar}┃ {n_fmt}/{total_fmt} "
                "• {elapsed}<{remaining} • {rate_fmt} • {postfix}"
            ),
        )
        for images, targets in progress:
            images = images.to(device)
            targets = targets.to(device)
 
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += targets.size(0)
 
            progress.set_postfix(
                loss=f"{running_loss / total_train:.4f}",
                acc=f"{correct_train / total_train:.2%}",
            )
 
        scheduler.step()
        train_loss = running_loss / len(train_ds)
        train_acc = correct_train / total_train
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
 
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
 
        print(
            f"Епоха  {epoch:02d}/{EPOCHS:02d}     | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc * 100:.2f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc * 100:.2f}%)"
        )
 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "val_accuracy": best_val_acc,
                },
                MODEL_PATH,
            )
 
    print(f"\nНайкраща модель збережена у: {MODEL_PATH} (Val Acc: {best_val_acc * 100:.2f}%)")
    save_training_curves(history, LOSS_PLOT_PATH)

    # ---------------- Фінальна оцінка на тестовій вибірці ----------------
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
 
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    cm = confusion_matrix(y_true, y_pred)
 
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТИ ЯКОСТІ МОДЕЛІ НА ТЕСТОВІЙ ВИБІРЦІ (TEST SPLIT)")
    print("=" * 60)
    print(f"Accuracy        : {test_acc * 100:.2f}%")
    print(f"Macro Precision : {precision * 100:.2f}%")
    print(f"Macro Recall    : {recall * 100:.2f}%")
    print(f"Macro F1-Score  : {f1 * 100:.2f}%")
    print(f"Macro ROC-AUC   : {roc_auc * 100:.2f}%")
    print("\nДетальний звіт по класах (Classification Report):")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
    print("Матриця помилок (Confusion Matrix):")
    print(cm)

    save_confusion_matrix(y_true, y_pred, CLASS_NAMES, CM_PLOT_PATH)
 
    metrics_summary = {
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "macro_roc_auc": float(roc_auc),
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "history": history,
    }
    METRICS_PATH.write_text(json.dumps(metrics_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Метрики збережено у: {METRICS_PATH}")


if __name__ == "__main__":
    main()
