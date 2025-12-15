# RealScape - Генератор реалистичных ландшафтов

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Диффузионная модель для генерации высокореалистичных игровых ландшафтов на основе Diffusion Models.
Интеграция в Unity с поддержкой CUDA.

## Быстрый старт

### Требования
- Python 3.10+
- CUDA 12.1+ (опционально)
- Git

### Установка

```bash
git clone https://github.com/YOUR_USERNAME/realscape.git
cd realscape
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Обучение

```bash
python training/train.py --device cuda --config training/config.yaml
```

### Генерация

```bash
python -c "from inference.generator import TerrainGenerator; gen = TerrainGenerator('models/checkpoints/model.pt'); heightmap = gen.generate()"
```

### API сервер

```bash
uvicorn api.server:app --reload
```

## Структура проекта

```
realscape/
├── models/              # Архитектуры UNet
├── preprocessing/       # Обработка OSM и высот
├── training/            # Обучение
├── inference/           # Генерация
├── api/                 # FastAPI сервер
├── unity_stub/          # C# скрипты для Unity
├── data/                # Датасеты
└── README.md
```


