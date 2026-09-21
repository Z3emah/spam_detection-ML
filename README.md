# SMS Spam Detector

A machine learning project that detects whether an SMS message is spam or legitimate using a trained scikit-learn pipeline and a lightweight Streamlit interface.

## Overview

This project includes:
- a Jupyter notebook for model development and experimentation
- a saved trained model (`best_pipeline.pkl`)
- a Streamlit application for live message classification
- evaluation metrics stored in `model_metrics.json`

The app allows users to either type a message or upload a `.txt` file and classify it as spam or not spam.

## Features

- SMS spam classification using a trained ML pipeline
- Real-time prediction through a user-friendly Streamlit UI
- Adjustable classification threshold
- File upload support for text-based input
- Model performance metrics included in the repository

## Project Structure

```text
spam_detection-ML/
├── RFC_SpamDetection.ipynb     # Model training and experimentation notebook
├── spamDetection.py            # Streamlit web application
├── best_pipeline.pkl           # Trained ML pipeline
├── model_metrics.json          # Evaluation metrics
├── requirements.txt            # Python dependencies
├── test_case(Temu).txt         # Example input for testing
├── test_case_2(Discord security code).txt
└── README.md                  # Project documentation
```

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Streamlit
- Joblib

## Setup

1. Clone the repository:

```bash
git clone https://github.com/Z3emah/spam_detection-ML.git
cd spam_detection-ML
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run spamDetection.py
```

Then open the local URL shown in the terminal (typically `http://localhost:8501`).

## Usage

- Enter an SMS message in the text area, or upload a text file.
- Click the `Classify` button.
- The model predicts whether the message is spam or not spam.
- You can adjust the classification threshold from the sidebar.

## Model Notes

The project uses a trained classification pipeline saved as `best_pipeline.pkl` and reports model metrics in `model_metrics.json`. The model is intended for educational/demo use and should not be treated as a production-grade spam filter.

## Disclaimer

This project is a machine learning demonstration for educational purposes only. It is not a professional spam filtering system.

## License

This project does not include a license file. If you plan to reuse or distribute the code, make sure to confirm the repository owner's licensing terms before using it in production.

## Author

Z3emah
