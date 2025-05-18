# Sales Prediction using Prophet

This project implements time series forecasting for sales data using Facebook's Prophet model. The model predicts sales for multiple items up to December 31, 2017.

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Files Required

The following data files should be placed in the project directory:
- `calendar.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sell_prices.csv`

## Usage

Run the prediction script:
```bash
python pomterm.py
```

The script will:
1. Load and process the sales data
2. Train Prophet models for each item
3. Generate predictions up to December 31, 2017
4. Save results to `prediction_df.csv`

## Features

- Handles multiple items simultaneously
- Includes US holidays in the model
- Supports yearly, weekly, and daily seasonality
- Progress tracking and checkpointing
- Memory optimization for large datasets

## Output

The script generates a CSV file (`prediction_df.csv`) containing:
- Date column (`ds`)
- Predicted sales values for each item
- Predictions from 2011-01-29 to 2017-12-31 