# Automated Time Series Forecasting System
![Stars](https://img.shields.io/github/stars/MateoOrtiz001/automated-time-series-forecasting-system?style=social)
![Watchers](https://img.shields.io/github/watchers/MateoOrtiz001/automated-time-series-forecasting-system?style=social)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)
![Version](https://img.shields.io/badge/Version-1.0-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

This repository contains an implementation of a TFT (Temporal Fusion Transformer) focused on predicting inflation series in Colombia. You can see the dashboard about data, model and some useful information at [this link](https://automated-time-series-forecasting-system.streamlit.app/).

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Automation](#-automation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [License](#-license)

## Features

- **Temporal Fusion Transformer (TFT)** model for inflation prediction
- **Automated data extraction** from multiple sources (BanRep, FAO, FRED)
- **Monthly pipeline** with automatic fine-tuning every 3 months
- **GitHub Actions automation** for scheduled monthly updates
- **Interactive dashboard** built with Streamlit
- **File rotation system** to maintain only the 2 most recent versions
- **Quantile predictions** with confidence intervals

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/MateoOrtiz001/automated-time-series-forecasting-system.git
cd automated-time-series-forecasting-system
```

### Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Run the Dashboard

```bash
streamlit run src/webApp/app.py
```

The dashboard will be available at `http://localhost:8501`

### Run the Monthly Pipeline

```bash
# Full pipeline (download + predict + cleanup)
python -m src.pipeline.monthly_pipeline

# Download data only
python -m src.pipeline.monthly_pipeline --download-only

# Predict only (without downloading new data)
python -m src.pipeline.monthly_pipeline --predict-only

# Force model fine-tuning
python -m src.pipeline.monthly_pipeline --finetune

# Clean old files only
python -m src.pipeline.monthly_pipeline --cleanup-only
```

### Use the Orchestrator

```bash
# Full pipeline
python -m src.pipeline.orchestrator --mode full

# Check system status
python -m src.pipeline.orchestrator --mode status

# Train model from scratch
python -m src.pipeline.orchestrator --mode train --epochs 200

# Run predictions
python -m src.pipeline.orchestrator --mode predict
```

## Automation

The system includes a GitHub Actions workflow that automatically runs the pipeline every month.

### How it works

```
┌─────────────────────────────────────────────────────────────────┐
│              AUTOMATED MONTHLY WORKFLOW                          │
├─────────────────────────────────────────────────────────────────┤
│  Day 5 of each month (10:00 UTC):                               │
│                                                                  │
│  1. GitHub Actions triggers the workflow                        │
│  2. Pipeline downloads data from BanRep, FAO, FRED              │
│  3. Data is consolidated into latest.csv                        │
│  4. Model generates predictions for next 12 months              │
│  5. Results are committed and pushed to the repository          │
│  6. Streamlit Cloud detects the push and re-deploys             │
│  7. Dashboard shows updated data automatically                  │
└─────────────────────────────────────────────────────────────────┘
```

### Manual trigger

You can also run the pipeline manually from the Actions tab:

1. Go to **Actions** > **Monthly Pipeline**
2. Click **Run workflow**
3. Optionally check "Force model fine-tuning"
4. Click **Run workflow**

## Project Structure

```
├── .github/
│   └── workflows/
│       └── monthly-pipeline.yml  # GitHub Actions workflow
├── data/
│   ├── raw/                    # Raw data from sources (not versioned)
│   │   ├── banrep/suameca/     # BanRep JSON files
│   │   └── external/           # External data (Brent, FAO)
│   └── proc/
│       └── latest.csv          # Current data for dashboard
├── misc/
│   ├── models/                 # Fine-tuned models
│   ├── results/
│   │   └── predictions_latest.csv  # Current predictions
│   └── logs/                   # Execution logs
├── models/                     # Base trained models
├── results/                    # Analysis results
├── src/
│   ├── etl/
│   │   └── dataExtractor.py    # Data extraction functions
│   ├── model/
│   │   ├── model.py            # TFT model implementation
│   │   └── customLayers.py     # Custom Keras layers
│   ├── pipeline/
│   │   ├── __init__.py         # Pipeline module exports
│   │   ├── core.py             # Core functions and configuration
│   │   ├── orchestrator.py     # High-level orchestrator (class)
│   │   ├── monthly_pipeline.py # CLI script for scheduled runs
│   │   └── pipeline_state.json # Pipeline state persistence
│   └── webApp/
│       └── app.py              # Streamlit dashboard
├── requirements.txt
├── Dockerfile
└── README.md
```

## Usage

### Data Sources

The system automatically downloads data from:

| Source | Data | Frequency |
|--------|------|-----------|
| **BanRep** | Inflation, GDP, Interest Rate, TRM, IPP | Monthly |
| **FAO** | Food Price Index | Monthly |
| **FRED** | Brent Oil Price | Monthly |

### Model Fine-tuning

The pipeline automatically performs fine-tuning every 3 months. You can force it with:

```bash
python -m src.pipeline.monthly_pipeline --finetune
```

### File Rotation System

The system automatically maintains only the 2 most recent versions of:
- Processed data files (`data/proc/*.csv`)
- Fine-tuned models (`misc/models/tft_finetuned_*.keras`)
- Predictions (`misc/results/predictions_*.csv`)
- Raw data files per series

## Contributing

We welcome contributions! Here's how you can help:

### Reporting Bugs

1. Check if the issue already exists in [Issues](https://github.com/MateoOrtiz001/automated-time-series-forecasting-system/issues)
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version)

### Suggesting Enhancements

1. Open an issue with the tag `enhancement`
2. Describe the feature and its use case
3. Include examples if possible

### Pull Requests

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes following the code style
4. Write/update tests if applicable
5. Commit with clear messages:
   ```bash
   git commit -m "feat: add new feature description"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a Pull Request with:
   - Description of changes
   - Related issue (if any)
   - Screenshots (for UI changes)

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and small

### Ideas for Contributions

- Add new data sources (other central banks, economic indicators)
- Implement additional models (LSTM, Prophet, N-BEATS)
- Improve dashboard visualizations
- Add more documentation
- Write unit tests
- Add support for other countries' inflation data
- Improve Docker configuration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **GitHub Issues**: For bugs and feature requests
- **Pull Requests**: For contributions

---

⭐ If you find this project useful, please consider giving it a star!

