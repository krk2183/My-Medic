"""
Medical Surge Prediction Interface
=================================

Easy-to-use interface for predicting medical condition surges
using the trained PyTorch model.

Usage:
    python prediction_interface.py --date "2024-03-01" --days 14
    python prediction_interface.py --date "2024-06-15" --days 30
"""

import argparse
import pandas as pd
import numpy as np
import torch
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from medical_surge_predictor import MedicalSurgeAnalyzer

def predict_surges_for_date_range(model_path, start_date, days_ahead):
    """
    Predict surges for a date range and return structured results
    
    Args:
        model_path: Path to the trained model (.pth file)
        start_date: Start date for predictions (string)
        days_ahead: Number of days to predict
    
    Returns:
        Dictionary with predictions and analysis
    """
    # Load the model
    analyzer = MedicalSurgeAnalyzer('overcrowding.csv')
    analyzer.load_model(model_path)
    
    # Make predictions
    predictions = analyzer.predict_surges(start_date, days_ahead)
    
    # Analyze predictions
    analysis = {
        'summary': {},
        'daily_predictions': predictions,
        'high_risk_periods': {},
        'seasonal_insights': {}
    }
    
    # Find high-risk conditions for each date
    for date, preds in predictions.items():
        high_risk = [(condition, prob) for condition, prob in preds.items() if prob > 0.3]
        high_risk.sort(key=lambda x: x[1], reverse=True)
        analysis['high_risk_periods'][date] = high_risk[:5]
    
    # Overall condition risk summary
    condition_totals = {}
    for date, preds in predictions.items():
        for condition, prob in preds.items():
            if condition not in condition_totals:
                condition_totals[condition] = []
            condition_totals[condition].append(prob)
    
    for condition, probs in condition_totals.items():
        analysis['summary'][condition] = {
            'avg_probability': np.mean(probs),
            'max_probability': np.max(probs),
            'high_risk_days': sum(1 for p in probs if p > 0.3)
        }
    
    # Seasonal insights based on the date range
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    analysis['seasonal_insights'] = {
        'period': f"{start_dt.strftime('%B %Y')} - {(start_dt + timedelta(days=days_ahead-1)).strftime('%B %Y')}",
        'season': get_season(start_dt.month),
        'high_risk_conditions': sorted(
            [(k, v['avg_probability']) for k, v in analysis['summary'].items()],
            key=lambda x: x[1], reverse=True
        )[:10]
    }
    
    return analysis

def get_season(month):
    """Get season name from month number"""
    seasons = {
        (12, 1, 2): 'Winter',
        (3, 4, 5): 'Spring',
        (6, 7, 8): 'Summer',
        (9, 10, 11): 'Fall'
    }
    for months, season in seasons.items():
        if month in months:
            return season
    return 'Unknown'

def create_prediction_visualization(analysis, output_file='surge_predictions.png'):
    """Create visualization of surge predictions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top conditions by average risk
    top_conditions = analysis['seasonal_insights']['high_risk_conditions'][:10]
    conditions, avg_probs = zip(*top_conditions)
    
    ax1.barh(range(len(conditions)), avg_probs, color='skyblue')
    ax1.set_yticks(range(len(conditions)))
    ax1.set_yticklabels([c[:20] + '...' if len(c) > 20 else c for c in conditions])
    ax1.set_xlabel('Average Surge Probability')
    ax1.set_title('Top 10 Conditions by Average Risk')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Daily risk timeline for top 5 conditions
    dates = list(analysis['daily_predictions'].keys())
    top_5_conditions = [c[0] for c in top_conditions[:5]]
    
    for condition in top_5_conditions:
        daily_probs = [analysis['daily_predictions'][date][condition] for date in dates]
        ax2.plot(range(len(dates)), daily_probs, marker='o', label=condition[:15])
    
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Surge Probability')
    ax2.set_title('Daily Risk Timeline (Top 5 Conditions)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(alpha=0.3)
    
    # 3. High-risk days count
    high_risk_counts = []
    for date in dates:
        count = len([p for p in analysis['daily_predictions'][date].values() if p > 0.3])
        high_risk_counts.append(count)
    
    ax3.bar(range(len(dates)), high_risk_counts, color='salmon')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Number of High-Risk Conditions')
    ax3.set_title('High-Risk Conditions per Day (>30% probability)')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Risk distribution
    all_probs = []
    for date in dates:
        all_probs.extend(analysis['daily_predictions'][date].values())
    
    ax4.hist(all_probs, bins=30, color='lightgreen', alpha=0.7, edgecolor='black')
    ax4.axvline(0.3, color='red', linestyle='--', label='High Risk Threshold')
    ax4.set_xlabel('Surge Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Surge Probabilities')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_file}")
    
    return fig

def generate_report(analysis, output_file='surge_report.txt'):
    """Generate a text report of the analysis"""
    with open(output_file, 'w') as f:
        f.write("MEDICAL SURGE PREDICTION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Prediction Period: {analysis['seasonal_insights']['period']}\n")
        f.write(f"Season: {analysis['seasonal_insights']['season']}\n\n")
        
        f.write("TOP 10 HIGH-RISK CONDITIONS:\n")
        f.write("-" * 30 + "\n")
        for i, (condition, prob) in enumerate(analysis['seasonal_insights']['high_risk_conditions'][:10], 1):
            f.write(f"{i:2d}. {condition:<30} {prob:.3f}\n")
        
        f.write("\nDAILY HIGH-RISK PERIODS:\n")
        f.write("-" * 30 + "\n")
        for date, high_risk in analysis['high_risk_periods'].items():
            if high_risk:  # Only show dates with high-risk conditions
                f.write(f"\n{date}:\n")
                for condition, prob in high_risk:
                    f.write(f"  • {condition:<25} {prob:.3f}\n")
        
        f.write("\nCONDITION SUMMARY STATISTICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Condition':<30} {'Avg Risk':<10} {'Max Risk':<10} {'High Days':<10}\n")
        f.write("-" * 60 + "\n")
        
        sorted_conditions = sorted(
            analysis['summary'].items(), 
            key=lambda x: x[1]['avg_probability'], 
            reverse=True
        )
        
        for condition, stats in sorted_conditions[:15]:
            f.write(f"{condition[:29]:<30} {stats['avg_probability']:.3f}      {stats['max_probability']:.3f}      {stats['high_risk_days']:<10}\n")
    
    print(f"Report saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Medical Surge Prediction Interface')
    parser.add_argument('--date', required=True, help='Start date for prediction (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=14, help='Number of days to predict (default: 14)')
    parser.add_argument('--model', default='medical_surge_model.pth', help='Path to model file')
    parser.add_argument('--output', default='predictions', help='Output file prefix')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--report', action='store_true', help='Generate text report')
    
    args = parser.parse_args()
    
    print(f"Predicting medical surges from {args.date} for {args.days} days...")
    
    try:
        # Make predictions
        analysis = predict_surges_for_date_range(args.model, args.date, args.days)
        
        # Save JSON output
        json_file = f"{args.output}_predictions.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"Predictions saved as {json_file}")
        
        # Print summary to console
        print("\nTOP 5 HIGH-RISK CONDITIONS:")
        print("-" * 40)
        for condition, prob in analysis['seasonal_insights']['high_risk_conditions'][:5]:
            print(f"{condition:<30} {prob:.3f}")
        
        print(f"\nSeason: {analysis['seasonal_insights']['season']}")
        print(f"Period: {analysis['seasonal_insights']['period']}")
        
        # Generate visualization if requested
        if args.visualize:
            create_prediction_visualization(analysis, f"{args.output}_chart.png")
        
        # Generate report if requested
        if args.report:
            generate_report(analysis, f"{args.output}_report.txt")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model file exists and the date format is correct (YYYY-MM-DD)")

if __name__ == "__main__":
    main()