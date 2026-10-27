"""
Задача 3: Веб-сервіс класифікації медичних знімків органів (FastAPI).

Забезпечує:
  - Зручну веб-сторінку (GET /) для завантаження знімка лікарем у браузері
  - REST API ендпоінт (POST /predict) для інтеграції з клінічними системами
  - Перевірку стану сервісу (GET /health)
"""

import io
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from train import build_model, MODEL_PATH
from dataset import CLASS_NAMES

app = FastAPI(
    title="Medical Organ Classification Service",
    description="Класифікація знімків органів (liver, kidney, lung) з оцінкою впевненості та маршрутизацією на онкоскринінг",
    version="1.0.0",
)

ORGAN_DESCRIPTIONS_UA = {
    "liver": {
        "name_ua": "Печінка (Liver)",
        "next_step": "Маршрутизація на ML-модель онкоскринінгу печінки (виявлення ГЦК / метастазів, пайплайн LiTS).",
    },
    "kidney": {
        "name_ua": "Нирки (Kidney)",
        "next_step": "Маршрутизація на ML-модель онкоскринінгу нирок (виявлення нирково-клітинного раку, пайплайн KiTS).",
    },
    "lung": {
        "name_ua": "Легені (Lung)",
        "next_step": "Маршрутизація на ML-модель онкоскринінгу легень (виявлення вузликів та новоутворень, пайплайн LUNA16/LIDC).",
    },
    "other": {
        "name_ua": "Інший орган (нецільовий: серце / селезінка / підшлункова)",
        "next_step": "Знімок не містить жодного з 3-х цільових органів (печінка, нирки, легені). Подальший онкоскринінг відхилено.",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODEL = None
LOADED_CLASSES = CLASS_NAMES


@app.on_event("startup")
def load_trained_model():
    global MODEL, LOADED_CLASSES
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        LOADED_CLASSES = checkpoint.get("class_names", CLASS_NAMES)
        MODEL = build_model(num_classes=len(LOADED_CLASSES), pretrained=False).to(DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        MODEL.eval()
        print(f"[INFO] Ваги моделі завантажено з {MODEL_PATH}")
    else:
        MODEL = build_model(num_classes=len(LOADED_CLASSES), pretrained=False).to(DEVICE)
        MODEL.eval()
        print(f"[WARN] Файл {MODEL_PATH} не знайдено! Сервіс запущено з неініціалізованими вагами (спочатку запустіть train.py).")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "weights_loaded": MODEL_PATH.exists(),
        "classes": LOADED_CLASSES,
    }


@app.post("/predict")
async def predict_organ(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл має бути зображенням (PNG, JPEG, BMP, TIFF).")

    raw_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("L").convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не вдалося прочитати зображення: {exc}")

    input_tensor = INFER_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

    best_idx = int( max(range(len(probs)), key=lambda i: probs[i]) )
    pred_class = LOADED_CLASSES[best_idx]
    confidence = round(probs[best_idx] * 100, 2)

    probabilities = {
        cls_name: round(prob * 100, 2)
        for cls_name, prob in zip(LOADED_CLASSES, probs)
    }

    organ_info = ORGAN_DESCRIPTIONS_UA.get(pred_class, {"name_ua": pred_class, "next_step": ""})

    return {
        "filename": file.filename,
        "predicted_class": pred_class,
        "organ_ua": organ_info["name_ua"],
        "confidence_percent": confidence,
        "probabilities_percent": probabilities,
        "recommended_next_step": organ_info["next_step"],
        "weights_loaded": MODEL_PATH.exists(),
    }


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Класифікація медичних знімків органів</title>
        <style>
            :root {
                --primary: #2563eb;
                --primary-hover: #1d4ed8;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text: #0f172a;
                --muted: #64748b;
                --border: #e2e8f0;
                --success: #16a34a;
                --danger: #dc2626;
                --highlight: #eff6ff;
            }

            * { box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 40px 16px;
                display: flex;
                justify-content: center;
            }

            .container {
                width: 100%;
                max-width: 760px;
            }

            .card {
                background: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 16px;
                padding: 32px;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
            }

            h1 {
                font-size: 24px;
                font-weight: 700;
                margin-top: 0;
                margin-bottom: 8px;
                color: #1e293b;
            }

            .subtitle {
                color: var(--muted);
                font-size: 15px;
                margin-bottom: 24px;
                line-height: 1.5;
            }

            .upload-zone {
                border: 2px dashed #cbd5e1;
                border-radius: 12px;
                padding: 24px;
                text-align: center;
                cursor: pointer;
                background-color: #fcfdfe;
                transition: all 0.2s ease;
            }
            .upload-zone:hover {
                border-color: var(--primary);
                background-color: #f8faff;
            }

            input[type="file"] {
                display: none;
            }

            .upload-label {
                cursor: pointer;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
            }

            .upload-icon {
                font-size: 36px;
            }

            .btn-submit {
                display: block;
                width: 100%;
                margin-top: 20px;
                background: var(--primary);
                color: white;
                border: none;
                padding: 13px 20px;
                border-radius: 10px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: background 0.2s ease, transform 0.1s ease;
            }
            .btn-submit:hover:not(:disabled) {
                background: var(--primary-hover);
            }
            .btn-submit:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            /* Прев'ю фото */
            .preview-container {
                display: none;
                margin-top: 20px;
                text-align: center;
                position: relative;
            }
            .preview-image {
                max-width: 100%;
                max-height: 280px;
                border-radius: 10px;
                border: 1px solid var(--border);
                box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
                object-fit: contain;
                background-color: #000;
            }
            .preview-title {
                font-size: 13px;
                font-weight: 600;
                color: var(--muted);
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            /* Результати */
            .results-card {
                margin-top: 28px;
                border-top: 1px solid var(--border);
                padding-top: 24px;
            }

            .result-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 16px 20px;
                border-radius: 12px;
                background-color: var(--highlight);
                border: 1px solid #bfdbfe;
                margin-bottom: 20px;
            }

            .result-title {
                font-size: 14px;
                color: #1e40af;
                margin-bottom: 4px;
            }

            .result-pred {
                font-size: 20px;
                font-weight: 800;
                color: #1e3a8a;
            }

            .result-badge {
                font-size: 18px;
                font-weight: 800;
                background: #2563eb;
                color: white;
                padding: 6px 14px;
                border-radius: 20px;
            }
            .result-badge.other {
                background: #475569;
            }

            /* Список результатів */
            .info-list {
                list-style: none;
                padding: 0;
                margin: 0;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            .info-item {
                display: flex;
                flex-direction: column;
                padding: 12px 16px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }

            .info-label {
                font-size: 13px;
                color: var(--muted);
                margin-bottom: 4px;
            }

            .info-value {
                font-size: 15px;
                color: var(--text);
            }

            /* Прогрес-бари ймовірностей */
            .prob-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 8px;
                font-size: 14px;
            }
            .prob-bar-bg {
                flex-grow: 1;
                margin: 0 14px;
                height: 9px;
                background: #e2e8f0;
                border-radius: 6px;
                overflow: hidden;
            }
            .prob-bar-fill {
                height: 100%;
                background: var(--primary);
                border-radius: 6px;
                transition: width 0.4s ease;
            }
            .prob-row.active .prob-name {
                font-weight: 700;
                color: #1e293b;
            }
            .prob-row.active .prob-val {
                font-weight: 700;
                color: var(--primary);
            }

            .spinner {
                display: inline-block;
                width: 18px;
                height: 18px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 0.8s linear infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>Діагностичний сервіс класифікації органів</h1>
                <div class="subtitle">
                    Автоматичне визначення органу на МРТ-знімку (<b>Печінка</b>, <b>Нирки</b>, <b>Легені</b> або <b>Інший орган</b>) з маршрутизацією на скринінг.
                </div>

                <div class="upload-zone" onclick="document.getElementById('imageInput').click()">
                    <label class="upload-label">
                        <span class="upload-icon">📁</span>
                        <span style="font-weight: 600; font-size: 15px;" id="uploadPrompt">Натисніть або перетягніть МРТ-знімок сюди</span>
                        <span style="font-size: 13px; color: var(--muted);">Формати: PNG, JPG, DICOM-експорт (довільний розмір або 224x224)</span>
                    </label>
                    <input type="file" id="imageInput" accept="image/*" onchange="handleFileSelect(event)" />
                </div>

                <div class="preview-container" id="previewContainer">
                    <div class="preview-title">Завантажений знімок</div>
                    <img id="previewImage" class="preview-image" alt="Знімок" />
                </div>

                <button id="submitBtn" class="btn-submit" onclick="sendImage()">Класифікувати знімок</button>

                <div id="resultsWrapper" style="display: none;" class="results-card">
                    <!-- Заповнюється скриптом -->
                </div>
            </div>
        </div>

        <script>
            let selectedFile = null;

            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (!file) return;
                selectedFile = file;

                document.getElementById('uploadPrompt').innerText = `Обрано: ${file.name}`;

                // Показуємо прев'ю завантаженого фото
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.getElementById('previewImage');
                    img.src = e.target.result;
                    document.getElementById('previewContainer').style.display = 'block';
                };
                reader.readAsDataURL(file);
            }

            async function sendImage() {
                if (!selectedFile) {
                    alert('Будь ласка, спочатку оберіть або перетягніть файл знімка!');
                    return;
                }

                const btn = document.getElementById('submitBtn');
                const resultsWrapper = document.getElementById('resultsWrapper');

                btn.disabled = true;
                btn.innerHTML = '<span class="spinner"></span> Аналіз знімка нейромережею...';

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const resp = await fetch('/predict', { method: 'POST', body: formData });
                    if (!resp.ok) {
                        const err = await resp.json();
                        throw new Error(err.detail || 'Помилка під час обробки');
                    }
                    const data = await resp.json();
                    renderStylishResults(data);
                } catch (err) {
                    alert('Помилка: ' + err.message);
                } finally {
                    btn.disabled = false;
                    btn.innerText = 'Класифікувати знімок';
                }
            }

            function renderStylishResults(data) {
                const resultsWrapper = document.getElementById('resultsWrapper');
                const isOther = data.predicted_class === 'other';

                // Назви органів для гарного відображення прогрес-барів
                const labelsMap = {
                    "liver": "Печінка (Liver)",
                    "kidney": "Нирки (Kidney)",
                    "lung": "Легені (Lung)",
                    "other": "Інше (Other)"
                };

                let probsHtml = '';
                for (const [cls, prob] of Object.entries(data.probabilities_percent)) {
                    const isActive = cls === data.predicted_class;
                    const displayName = labelsMap[cls] || cls;
                    probsHtml += `
                        <div class="prob-row ${isActive ? 'active' : ''}">
                            <span class="prob-name" style="width: 140px;">${isActive ? '<b>' + displayName + '</b>' : displayName}</span>
                            <div class="prob-bar-bg">
                                <div class="prob-bar-fill" style="width: ${prob}%; ${isActive ? (isOther ? 'background: #64748b;' : 'background: #2563eb;') : ''}"></div>
                            </div>
                            <span class="prob-val" style="width: 60px; text-align: right;">${isActive ? '<b>' + prob + '%</b>' : prob + '%'}</span>
                        </div>
                    `;
                }

                resultsWrapper.innerHTML = `
                    <div class="result-header" style="${isOther ? 'background-color: #f1f5f9; border-color: #cbd5e1;' : ''}">
                        <div>
                            <div class="result-title" style="${isOther ? 'color: #475569;' : ''}">Визначений орган:</div>
                            <div class="result-pred" style="${isOther ? 'color: #334155;' : ''}"><b>${data.organ_ua}</b></div>
                        </div>
                        <div class="result-badge ${isOther ? 'other' : ''}">
                            <b>${data.confidence_percent}%</b>
                        </div>
                    </div>

                    <ul class="info-list">
                        <li class="info-item">
                            <span class="info-label">Назва файлу:</span>
                            <span class="info-value"><b>${data.filename}</b></span>
                        </li>
                        <li class="info-item">
                            <span class="info-label">Розподіл ймовірностей по класах:</span>
                            <div style="margin-top: 4px;">
                                ${probsHtml}
                            </div>
                        </li>
                        <li class="info-item" style="border-left: 4px solid ${isOther ? '#94a3b8' : '#2563eb'};">
                            <span class="info-label">Рекомендована клінічна дія:</span>
                            <span class="info-value" style="font-size: 14px; line-height: 1.4;">
                                ${data.recommended_next_step}
                            </span>
                        </li>
                    </ul>
                `;

                resultsWrapper.style.display = 'block';
                resultsWrapper.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        </script>
    </body>
    </html>
    """
