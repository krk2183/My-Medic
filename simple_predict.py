#!/usr/bin/env python3
"""
Simple Medical Surge Prediction Script
=====================================

Quick and easy way to get surge predictions for any date.

Usage:
    python simple_predict.py
    
Then follow the prompts to enter your desired prediction date and duration.
"""

from medical_surge_predictor import MedicalSurgeAnalyzer
from datetime import datetime, timedelta
import json

def main():
    print("🏥 Medical Surge Prediction System")
    print("=" * 40)
    
    # Load the model
    print("Loading trained model...")
    try:
        analyzer = MedicalSurgeAnalyzer('overcrowding.csv')
        analyzer.load_model('medical_surge_model.pth')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\n" + "-" * 40)
    
    # Get user input
    while True:
        try:
            date_input = input("Enter prediction start date (YYYY-MM-DD) or 'quit': ")
            if date_input.lower() == 'quit':
                break
                
            # Validate date format
            datetime.strptime(date_input, '%Y-%m-%d')
            
            days_input = input("Enter number of days to predict (1-365): ")
            days = int(days_input)
            
            if days < 1 or days > 365:
                print("Please enter a number between 1 and 365")
                continue
                
            break
            
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD for date and a number for days.")
            continue
    
    if date_input.lower() == 'quit':
        print("Goodbye!")
        return
    
    # Make predictions
    print(f"\n🔮 Predicting surges from {date_input} for {days} days...")
    print("Please wait...")
    
    try:
        predictions = analyzer.predict_surges(date_input, days_ahead=days)
        
        print("\n" + "=" * 50)
        print("📊 PREDICTION RESULTS")
        print("=" * 50)
        
        # Calculate overall statistics
        all_conditions = set()
        for date_preds in predictions.values():
            all_conditions.update(date_preds.keys())
        
        condition_stats = {}
        for condition in all_conditions:
            probs = [predictions[date][condition] for date in predictions.keys()]
            condition_stats[condition] = {
                'avg': sum(probs) / len(probs),
                'max': max(probs),
                'high_risk_days': sum(1 for p in probs if p > 0.3)
            }
        
        # Show top 10 highest risk conditions
        top_conditions = sorted(condition_stats.items(), key=lambda x: x[1]['avg'], reverse=True)[:10]
        
        print(f"\n🎯 TOP 10 HIGHEST RISK CONDITIONS (Average over {days} days):")
        print("-" * 60)
        print(f"{'Rank':<4} {'Condition':<30} {'Avg Risk':<10} {'Max Risk':<10} {'High Days'}")
        print("-" * 60)
        
        for i, (condition, stats) in enumerate(top_conditions, 1):
            print(f"{i:<4} {condition[:29]:<30} {stats['avg']:.3f}      {stats['max']:.3f}      {stats['high_risk_days']}")
        
        # Show daily high-risk periods
        print(f"\n⚠️  HIGH-RISK PERIODS (Probability > 0.30):")
        print("-" * 50)
        
        high_risk_found = False
        for date, date_preds in predictions.items():
            high_risk_conditions = [(cond, prob) for cond, prob in date_preds.items() if prob > 0.30]
            
            if high_risk_conditions:
                high_risk_found = True
                high_risk_conditions.sort(key=lambda x: x[1], reverse=True)
                print(f"\n📅 {date}:")
                for condition, prob in high_risk_conditions[:5]:  # Show top 5 per day
                    print(f"   • {condition:<25} {prob:.3f}")
        
        if not high_risk_found:
            print("   No high-risk periods detected (all probabilities < 0.30)")
        
        # Seasonal insight
        start_date = datetime.strptime(date_input, '%Y-%m-%d')
        season_map = {
            (12, 1, 2): '❄️  Winter',
            (3, 4, 5): '🌸 Spring',
            (6, 7, 8): '☀️  Summer',
            (9, 10, 11): '🍂 Fall'
        }
        
        season = 'Unknown'
        for months, season_name in season_map.items():
            if start_date.month in months:
                season = season_name
                break
        
        print(f"\n🌍 SEASONAL CONTEXT:")
        print(f"Season: {season}")
        print(f"Period: {start_date.strftime('%B %d, %Y')} - {(start_date + timedelta(days=days-1)).strftime('%B %d, %Y')}")
        
        # Save results
        output_file = f"predictions_{date_input}_{days}days.json"
        with open(output_file, 'w') as f:
            json.dump({
                'prediction_period': f"{date_input} to {(start_date + timedelta(days=days-1)).strftime('%Y-%m-%d')}",
                'season': season,
                'top_conditions': top_conditions,
                'daily_predictions': predictions,
                'statistics': condition_stats
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
    
    print("\n" + "=" * 50)
    print("Thank you for using the Medical Surge Prediction System!")
    print("=" * 50)

if __name__ == "__main__":
    main()