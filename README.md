# ChurnShield Pro: IBM Telco Analytics

ChurnShield Pro is a high-performance predictive analytics platform powered by the real-world **IBM Telco Customer Churn** dataset. Unlike synthetic models, ChurnShield Pro provides deep insights based on thousands of authentic customer profiles, making it production-ready for retention intelligence.

## Features

- **Real-World Intelligence**: Trained on 7,043 authentic customer records with 21 feature dimensions.
- **Deep Service Analysis**: Evaluates risk based on Internet type (Fiber vs DSL), Streaming habits, and Security services.
- **Optimized Preprocessing**: Implements a robust data pipeline that handles missing values and maps complex categorical relationships.
- **Enterprise UI**: A comprehensive, grouped dashboard for detailed customer risk audits.

## Data Schema (21 Features)
The model analyzes a full customer profile including:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents.
- **Service Stack**: Phone, Multiple Lines, Internet Service (Fiber/DSL/None).
- **Security & Support**: Online Security, Online Backup, Device Protection, Tech Support.
- **Entertainment**: Streaming TV, Streaming Movies.
- **Financials**: Tenure, Monthly Charges, Total Charges, Contract Type, Payment Method, Paperless Billing.

## Getting Started

### Prerequisites
- Python 3.8+
- Requirements: `pandas`, `numpy`, `scikit-learn`, `flask`, `joblib`

### Installation
1. Clone the repository.
2. Initialize and Fetch Data:
   ```bash
   python generate_data.py
   ```
3. Train Production Model:
   ```bash
   python train.py
   ```
4. Start Service:
   ```bash
   python app.py
   ```

## Model Performance
The current production model is a **Logistic Regression** pipeline, achieving an accuracy of **~79%** on the IBM Telco benchmark. It provides a balanced precision across both stable and at-risk segments.
