# Riballo Symmetry Axis Labeling Application
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
> A Python-based GUI tool for labeling symmetry axes in images and calculating
inter-rater agreement scores.
---
## Table of Contents
1. [Features](#-features)
2. [System Requirements](#-system-requirements)
3. [Project Structure](#-project-structure)
4. [Data Preparation](#-data-preparation)
5. [Configuration](#-configuration)
6. [Installation](#-installation)
7. [Usage](#-usage)
8. [Output Files](#-output-files)
9. [Troubleshooting and Notes](#-troubleshooting-and-notes)
---
## Features
- GUI with **tkinter**
- Supports `.mat` and `.csv` input formats
- Symmetry axis labeling with guided questions:
 - Acceptable? (Yes/No)
 - Principal? (Yes/No if acceptable)
- Auto-advance behavior
- Classification logic (YY, YN, NN)
- Image scaling & Y-axis inversion support
- Session data saved with metadata
- Calculates PA/NA scores across sessions
- Tutorial mode for new users
---
## System Requirements
- **OS:** Windows
- **Python:** 3.8+
- **Libraries:**
 ```bash
 pip install Pillow scipy numpy
 ```
- `tkinter` and `csv` are part of the Python Standard Library
---
## Project Structure
```text
your_application_root/
 src/
 main.py
 gui.py
 data_manager.py
 config.py
 data/
 images/
 mat_files/
 csv_files/
 results/
 score/
 README.md
 RSALA_Report.pdf
```
---
## Data Preparation
### Naming Convention
- Format: `refs_XXX` (e.g., `refs_001`)
- Match `N_IMAGES` in `config.py`
### Image Files
- Path: `data/images/`
- Format: `.jpg`
- Examples:
 ```text
 refs_001.jpg, refs_002.jpg, ...
 ```
### MAT Files
- Path: `data/mat_files/`
- Format: `refs_XXX.mat`
- Must include `img_detected_refs` variable:
 ```matlab
 [x1, y1, x2, y2, score]
 ```
### CSV Input Files
**Option A: Single CSV**
```csv
image_base_name,axis_row_index,x1,y1,x2,y2,score
refs_001,0,100.0,50.0,100.0,200.0,1.0
```
**Option B: Multiple CSVs**
- Format: `refs_XXX.csv`
```csv
axis_row_index,x1,y1,x2,y2,score
0,100.0,50.0,100.0,200.0,1.0
```
---
## Configuration
Edit `src/config.py` to set:
- `N_IMAGES`, `AXES_PER_IMAGE`, `QUESTIONS_PER_SESSION`, `TUTORIAL_MODE`
- Thresholds: `THRESHOLD_NEAR_ONE`, `THRESHOLD_FAR_FROM_ONE`
- Target class distribution
- Data paths
---
## Installation
```bash
git clone https://github.com/yourusername/riballo-symmetry-axis-labeling.git
cd riballo-symmetry-axis-labeling
pip install Pillow scipy numpy
```
Prepare the data folders and adjust `src/config.py` as needed.
---
## Usage
### Launch the App
```bash
python src/main.py
```
### Workflow
1. Choose `.mat` or `.csv`
2. Choose single or multiple CSV files
3. Label axes using GUI:
 - Q1: Acceptable axis?
 - Q2: Principal axis?
4. Save session results
5. Calculate PA/NA scores
---
## Output Files
### Results (`results/`)
- CSV format
- Example filename:
 `session_results_20250725_141020_Q50_TYY40_TYN30.csv`
### Scores (`score/`)
- Appends to `score/score.csv`
- Requires at least 2 session results with matching configs
---
## Troubleshooting and Notes
- Use correct file paths (`/` or `\`)
- Follow exact file formats
- Scores won't calculate if configs mismatch
---
## References
- For PA/NA metric theory: [IEEE Paper](https://ieeexplore.ieee.org/document/9459467)
(Section VI)
---
