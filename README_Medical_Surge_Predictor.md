# Medical Surge Prediction System

A PyTorch-based machine learning system that predicts which medical conditions will experience surges on specific dates using historical hospital admission data.

## 🎯 Overview

This system analyzes temporal patterns in medical data to predict when specific conditions (like Heart Failure, AKI, UTI, etc.) will experience higher-than-normal patient volumes. It's designed for healthcare administrators and planners to anticipate resource needs.

## 📁 Files Generated

- **`medical_surge_predictor.py`** - Complete PyTorch model source code
- **`medical_surge_model.pth`** - Trained model weights and parameters
- **`prediction_interface.py`** - Easy-to-use prediction interface
- **`overcrowding.csv`** - Input dataset (your original data)

## 🏗️ Model Architecture

**Neural Network Design:**
- **Input Layer:** 11 temporal features (month, day of week, seasonality, etc.)
- **Hidden Layers:** 128 → 64 → 32 neurons with ReLU activation
- **Regularization:** Batch normalization and dropout (30%)
- **Output Layer:** 36 medical conditions with sigmoid activation
- **Loss Function:** Binary Cross Entropy (multi-label classification)

**Features Used:**
- Month, day of week, day of year
- Week of year, quarter, season
- Cyclical time encoding (sin/cos transforms)
- Average patient age

## 🧠 Model Capabilities

### Predicted Conditions (36 total):
- **Cardiovascular:** ACS, STEMI, Heart Failure, Atrial Fibrillation, VT, PSVT
- **Systemic:** Diabetes (DM), Hypertension (HTN), Acute Kidney Injury (AKI)
- **Respiratory:** Chest Infection, Pulmonary Embolism
- **Infections:** UTI, Infective Endocarditis
- **Neurological:** CVA Infarct, CVA Bleed, Neuro Cardiogenic Syncope
- **And 20+ more conditions**

### Seasonal Insights from Training Data:
```
Average Seasonal Condition Counts:
Winter: Higher rates of severe conditions
Spring: Moderate cardiovascular events  
Summer: Lower overall admission rates
Fall:   Increased chronic condition exacerbations
```

## 🚀 Quick Start

### 1. Basic Prediction
```python
from medical_surge_predictor import MedicalSurgeAnalyzer

# Load trained model
analyzer = MedicalSurgeAnalyzer('overcrowding.csv')
analyzer.load_model('medical_surge_model.pth')

# Predict surges for next week
predictions = analyzer.predict_surges('2024-03-01', days_ahead=7)

# Show results
for date, conditions in predictions.items():
    print(f"Date: {date}")
    # Show top 3 highest risk conditions
    top_conditions = sorted(conditions.items(), key=lambda x: x[1], reverse=True)[:3]
    for condition, probability in top_conditions:
        print(f"  {condition}: {probability:.3f}")
```

### 2. Command Line Interface
```bash
# Predict surges for 2 weeks starting March 1st, 2024
python prediction_interface.py --date "2024-03-01" --days 14

# With visualization and report
python prediction_interface.py --date "2024-06-15" --days 30 --visualize --report

# Custom output prefix
python prediction_interface.py --date "2024-12-01" --days 7 --output "winter_surge"
```

### 3. Advanced Usage
```python
# Retrain model with new data
analyzer = MedicalSurgeAnalyzer('new_data.csv')
analyzer.load_and_preprocess_data()
X_train, X_test, y_train, y_test, _ = analyzer.prepare_training_data()
model = analyzer.build_model(X_train.shape[1], y_train.shape[1])
analyzer.train_model(X_train, y_train, X_test, y_test, epochs=100)
analyzer.save_model('updated_model.pth')
```

## 📊 Output Examples

### Console Output:
```
TOP 5 HIGH-RISK CONDITIONS:
PSVT                           0.316
ATYPICAL CHEST PAIN           0.260  
VT                            0.246
NEURO CARDIOGENIC SYNCOPE     0.238
SEVERE ANAEMIA                0.216

Season: Spring
Period: March 2024 - March 2024
```

### JSON Output (`predictions.json`):
```json
{
  "summary": {
    "AKI": {
      "avg_probability": 0.142,
      "max_probability": 0.198,
      "high_risk_days": 2
    }
  },
  "daily_predictions": {
    "2024-03-01": {
      "AKI": 0.142,
      "HEART FAILURE": 0.089,
      "UTI": 0.156
    }
  },
  "seasonal_insights": {
    "season": "Spring",
    "high_risk_conditions": [...]
  }
}
```

## 🎨 Visualization Features

When using `--visualize`, generates a comprehensive chart showing:
1. **Top 10 Conditions by Risk** - Bar chart of average probabilities
2. **Daily Timeline** - Risk trends over time for top 5 conditions  
3. **High-Risk Days** - Count of conditions above 30% probability per day
4. **Risk Distribution** - Histogram of all surge probabilities

## 📈 Model Performance

**Training Results:**
- Training Loss: 0.401 (final)
- Test Loss: 0.469 (final)
- Architecture: 11 → 128 → 64 → 32 → 36
- Training Samples: 667
- Test Samples: 167
- Conditions Predicted: 36

**Interpretation:**
- **Probability > 0.30:** High surge risk (requires attention)
- **Probability 0.15-0.30:** Moderate risk (monitor closely)
- **Probability < 0.15:** Low risk (normal levels expected)

## 🔧 Customization Options

### Modify Surge Definition:
```python
# In create_surge_data method, adjust threshold
surge_df = analyzer.create_surge_data(window_days=7, surge_threshold=1.5)
# surge_threshold=2.0 for more conservative surge detection
# window_days=14 for longer aggregation periods
```

### Add New Features:
```python
# In load_and_preprocess_data method, add:
self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
self.df['is_holiday'] = self.df['D.O.A'].dt.month.isin([12, 1]).astype(int)

# Update feature_columns list accordingly
```

## 📅 Use Cases

### Healthcare Administration:
- **Staffing Planning:** Predict busy periods for specific specialties
- **Resource Allocation:** Anticipate equipment and bed needs
- **Budget Planning:** Forecast high-cost procedure volumes

### Clinical Research:
- **Epidemiological Studies:** Understand seasonal disease patterns
- **Population Health:** Identify community health trends
- **Quality Improvement:** Plan interventions for high-risk periods

### Real-World Examples:
```python
# Winter flu season planning
predictions = analyzer.predict_surges('2024-12-01', days_ahead=90)

# Summer trauma season (accidents, heat-related)  
predictions = analyzer.predict_surges('2024-06-01', days_ahead=60)

# Back-to-school health surge
predictions = analyzer.predict_surges('2024-08-15', days_ahead=30)
```

## ⚠️ Important Notes

1. **Data Requirements:** Model trained on 2017 data; performance may vary with different time periods
2. **Surge Definition:** Currently defines surge as 1.5x historical average within 7-day windows
3. **Temporal Scope:** Best for short to medium-term predictions (days to months)
4. **External Factors:** Doesn't account for pandemics, policy changes, or external events

## 🔄 Model Updates

To retrain with new data:
1. Replace `overcrowding.csv` with new dataset (same format)
2. Run: `python medical_surge_predictor.py`
3. New `medical_surge_model.pth` will be generated

## 📋 Requirements

```
pandas>=2.0.0
numpy>=1.20.0
torch>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🎯 Key Features Summary

✅ **36 Medical Conditions** predicted simultaneously  
✅ **Seasonal Pattern Recognition** built into model architecture  
✅ **Custom Date Range** predictions (days to months ahead)  
✅ **Multiple Output Formats** (JSON, visualization, reports)  
✅ **Easy Integration** with existing healthcare systems  
✅ **Interpretable Results** with probability scores  
✅ **Retrainable** with new data  

---

*Built with PyTorch for robust, scalable medical surge prediction.*