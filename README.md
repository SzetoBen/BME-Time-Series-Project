# BME-Time-Series-Project
# Predicting Charlottesville Parking Ticket Appeal Success Using Time Series Analysis   

## Group  
**BME –  Masato Takedai (leader), Ben Szeto, Eddie Zhang**  
DS 4002 – 001 – 1pm – October 2025  

---

## Contents of Repository  
- **README.md** – Orientation and instructions for reproducing results  
- **LICENSE.md** – MIT License for reuse of this repository  
- **SCRIPTS/** – Source code for preprocessing and sentiment analysis  
- **DATA/** – Cleaned datasets, plus Data metadata 
- **OUTPUT/** – Figures, tables, and other analysis results  

---

## Software and Platform  
- **Programming Language:** Python 3.10+  
- **Core Libraries:**  
  - `pandas` – data manipulation  
  - `numpy` – numerical processing  
  - `matplotlib`, `seaborn` – visualizations  
  - `vaderSentiment` – lexicon-based sentiment analysis  
  - `scikit-learn` (sklearn) – building and evaulating machine learning models
  - `xgboost` - library for our SVM model 
- **Platform:** Code developed and tested on Windows and MacOS  

---

## Project Folder Map  
 ```
BME-Time-Series-Project

├── data
│   ├── Parking_Tickets.csv
│   ├── Processed_Parking_Tickets.csv
│   ├── Metadata.md
│   ├── Figure 1.png
│   └── Figure 2.png
├── LICENSE
├── output
│   ├── results_logistic.txt
│   ├── results_random_forest.txt
│   ├── results_xgboost.txt
│   ├── roc_logistic.png
│   ├── roc_random_forest.png
│   └── roc_xgboost.png
├── README.md
└── scripts
    ├── ParkingTicketsCleaning.ipynb
    └── train.py
```

---

## Instructions for Reproducing Results  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/SzetoBen/BME-Time-Series-Project
2. **Install dependencies**\
    pip install -r requirements.txt
3. **Prepare the data**\
    Use the cleaning notebook "ParkingTicketsCleaning.ipynb" in the scripts folder to clean the dataset. Resulting cleaned dataset will be in the data folder.
4. **Training models**
    1) **Train models using train.py**\
        1. 
    
1) https://datauvalibrary.opendata.arcgis.com/datasets/charlottesville::parking-tickets/about

