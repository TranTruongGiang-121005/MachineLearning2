# Music Genre Classification

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Audio data directory structure: `../genres_original/{genre}/*.wav`

### Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt** includes:
- `librosa` - Audio feature extraction
- `scikit-learn` - ML algorithms, preprocessing, evaluation
- `xgboost` - Gradient boosting
- `pandas`, `numpy` - Data handling
- `matplotlib`, `seaborn` - Visualization
- `joblib` - Model persistence
- `streamlit` - Presentation

## Running
### To create a new dataset based on genres_original folder
```bash
python3 generate_enhanced_database.py
```
### Train the model
```bash
python3 train2.py
```
### Run the streamlit app
```bash
streamlit run app.py
```