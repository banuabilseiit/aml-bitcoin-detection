# 🔍 AML Bitcoin Transaction Detection
### RegTech | Надзор за цифровыми активами

Проект по машинному обучению для автоматической детекции подозрительных транзакций в сети Bitcoin с позиции финансового регулятора.

---

## 📋 Описание проекта

**Бизнес-задача:** Автоматизация AML-скрининга (Anti-Money Laundering) криптовалютных транзакций в соответствии с требованиями FATF Recommendation 16.

**ML-задача:** Бинарная классификация — определение является ли транзакция подозрительной (illicit) или легитимной (licit).

**Датасет:** [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- 203,769 транзакций Bitcoin
- 166 признаков (локальные + агрегированные)
- 49 временных шагов
- 2 класса: illicit (подозрительная) / licit (легитимная)

---

## 🏗️ Структура проекта

```
├── aml_bitcoin_detection.ipynb   # Основной ноутбук
├── requirements.txt              # Python зависимости
├── Dockerfile                    # Docker для воспроизводимости
├── data/                         # Данные (не коммитятся в git)
│   └── elliptic_bitcoin_dataset/
└── README.md
```

---

## 🤖 Модели и результаты

| Модель | ROC-AUC | PR-AUC | F1 (illicit) |
|---|---|---|---|
| Logistic Regression | ~0.91 | ~0.72 | ~0.68 |
| Random Forest | ~0.96 | ~0.84 | ~0.80 |
| **XGBoost (tuned)** | **~0.98** | **~0.88** | **~0.85** |

---

## 📊 Содержание ноутбука

1. **Загрузка данных** — Kaggle API интеграция
2. **EDA** — дисбаланс классов, временная динамика, корреляции
3. **Feature Engineering** — временное разбиение train/test, SMOTE-балансировка
4. **Моделирование** — Logistic Regression, Random Forest, XGBoost
5. **Сравнение моделей** — ROC/PR-кривые, Confusion Matrix
6. **Hyperparameter Tuning** — RandomizedSearchCV
7. **Data Drift Monitoring** — PSI, KS-тест, деградация модели во времени
8. **SHAP интерпретация** — объяснение предсказаний
9. **Threshold Analysis** — выбор порога для AML
10. **RegTech выводы** — импликации для регулятора

---

## 🚀 Запуск

### Локально
```bash
git clone https://github.com/your-username/aml-bitcoin-detection
cd aml-bitcoin-detection
pip install -r requirements.txt

# Настройте Kaggle API (https://www.kaggle.com/settings → API → Create Token)
# Поместите kaggle.json в ~/.kaggle/

jupyter notebook aml_bitcoin_detection.ipynb
```

### Docker
```bash
docker build -t aml-detection .
docker run -p 8888:8888 -v $(pwd)/data:/app/data aml-detection
# Откройте http://localhost:8888
```

---

## 🏛️ RegTech-контекст

Проект демонстрирует применение ML-методов в задачах финансового надзора:

- **AML-скрининг** — автоматическая флагировка транзакций для ручной проверки
- **FATF Compliance** — соответствие требованиям Travel Rule (Rec. 16)
- **PSI-мониторинг** — отраслевой стандарт контроля стабильности данных в финансах
- **Порог классификации** — настройка под регуляторные требования (минимизация FN)
- **Интерпретируемость** — SHAP объяснения для комплаенс-офицеров

---

## 📦 Зависимости

- Python 3.10+
- scikit-learn, xgboost, imbalanced-learn
- shap, matplotlib, seaborn
- pandas, numpy, scipy

---

*Курс AI Engineer — Machine Learning Project*
