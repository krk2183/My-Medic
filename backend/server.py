from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime, timedelta
import sys
import json

# Add the parent directory to path to import our model
sys.path.append(str(Path(__file__).parent.parent))
from medical_surge_predictor import MedicalSurgeAnalyzer


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

# Medical Surge Prediction Models
class PredictionRequest(BaseModel):
    start_date: str
    days_ahead: int = 7

class PredictionResponse(BaseModel):
    predictions: Dict[str, Dict[str, float]]
    summary: Dict[str, Dict[str, float]]
    high_risk_periods: Dict[str, List[Dict[str, float]]]
    seasonal_insights: Dict

class MetricsResponse(BaseModel):
    total_conditions: int
    high_risk_conditions: int
    avg_risk_score: float
    peak_risk_date: str
    seasonal_trend: str

class ConditionFilterRequest(BaseModel):
    conditions: List[str]
    start_date: str
    days_ahead: int = 7

# Initialize the medical surge analyzer
medical_analyzer = None

def initialize_model():
    """Initialize the medical surge prediction model"""
    global medical_analyzer
    try:
        model_path = Path(__file__).parent.parent / "medical_surge_model.pth"
        data_path = Path(__file__).parent.parent / "overcrowding.csv"
        
        medical_analyzer = MedicalSurgeAnalyzer(str(data_path))
        medical_analyzer.load_model(str(model_path))
        logger.info("Medical surge prediction model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load medical model: {e}")
        return False

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Medical Surge Prediction API"}

# Medical Surge Prediction Endpoints
@api_router.post("/predict-surge", response_model=PredictionResponse)
async def predict_surge(request: PredictionRequest):
    """Predict medical condition surges for a given date range"""
    if not medical_analyzer:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Get predictions from the model
        predictions = medical_analyzer.predict_surges(
            request.start_date, 
            days_ahead=request.days_ahead
        )
        
        # Calculate summary statistics
        summary = {}
        all_conditions = set()
        for date_preds in predictions.values():
            all_conditions.update(date_preds.keys())
        
        for condition in all_conditions:
            probs = [predictions[date][condition] for date in predictions.keys()]
            summary[condition] = {
                'avg_probability': sum(probs) / len(probs),
                'max_probability': max(probs),
                'min_probability': min(probs),
                'high_risk_days': sum(1 for p in probs if p > 0.3)
            }
        
        # Find high-risk periods
        high_risk_periods = {}
        for date, preds in predictions.items():
            high_risk = [
                {'condition': condition, 'probability': prob} 
                for condition, prob in preds.items() if prob > 0.3
            ]
            if high_risk:
                high_risk_periods[date] = sorted(high_risk, key=lambda x: x['probability'], reverse=True)[:5]
        
        # Seasonal insights
        start_dt = datetime.strptime(request.start_date, '%Y-%m-%d')
        season_map = {
            (12, 1, 2): 'Winter',
            (3, 4, 5): 'Spring',
            (6, 7, 8): 'Summer',
            (9, 10, 11): 'Fall'
        }
        
        season = 'Unknown'
        for months, season_name in season_map.items():
            if start_dt.month in months:
                season = season_name
                break
        
        seasonal_insights = {
            'season': season,
            'period': f"{start_dt.strftime('%B %Y')} - {(start_dt + timedelta(days=request.days_ahead-1)).strftime('%B %Y')}",
            'top_conditions': sorted(
                [(k, v['avg_probability']) for k, v in summary.items()],
                key=lambda x: x[1], reverse=True
            )[:10]
        }
        
        return PredictionResponse(
            predictions=predictions,
            summary=summary,
            high_risk_periods=high_risk_periods,
            seasonal_insights=seasonal_insights
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.get("/metrics")
async def get_current_metrics():
    """Get current system metrics and model status"""
    if not medical_analyzer:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Get predictions for next 7 days
        today = datetime.now().strftime('%Y-%m-%d')
        predictions = medical_analyzer.predict_surges(today, days_ahead=7)
        
        # Calculate metrics
        all_probs = []
        high_risk_count = 0
        peak_risk_date = today
        max_daily_risk = 0
        
        for date, preds in predictions.items():
            daily_probs = list(preds.values())
            all_probs.extend(daily_probs)
            
            high_risk_today = sum(1 for p in daily_probs if p > 0.3)
            if high_risk_today > high_risk_count:
                high_risk_count = high_risk_today
                peak_risk_date = date
            
            daily_avg = sum(daily_probs) / len(daily_probs)
            if daily_avg > max_daily_risk:
                max_daily_risk = daily_avg
                peak_risk_date = date
        
        # Determine seasonal trend
        month = datetime.now().month
        seasonal_trend = {
            (12, 1, 2): 'High winter surge risk',
            (3, 4, 5): 'Moderate spring conditions',
            (6, 7, 8): 'Lower summer risk period',
            (9, 10, 11): 'Fall surge preparation needed'
        }
        
        trend = 'Monitoring required'
        for months, trend_text in seasonal_trend.items():
            if month in months:
                trend = trend_text
                break
        
        return MetricsResponse(
            total_conditions=len(medical_analyzer.condition_columns),
            high_risk_conditions=high_risk_count,
            avg_risk_score=sum(all_probs) / len(all_probs) if all_probs else 0,
            peak_risk_date=peak_risk_date,
            seasonal_trend=trend
        )
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")

@api_router.post("/filter-conditions")
async def filter_conditions(request: ConditionFilterRequest):
    """Get predictions filtered by specific conditions"""
    if not medical_analyzer:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Get all predictions
        predictions = medical_analyzer.predict_surges(
            request.start_date, 
            days_ahead=request.days_ahead
        )
        
        # Filter by requested conditions
        filtered_predictions = {}
        for date, preds in predictions.items():
            filtered_predictions[date] = {
                condition: prob for condition, prob in preds.items()
                if condition in request.conditions
            }
        
        return {"filtered_predictions": filtered_predictions}
        
    except Exception as e:
        logger.error(f"Filter error: {e}")
        raise HTTPException(status_code=500, detail=f"Filtering failed: {str(e)}")

@api_router.get("/available-conditions")
async def get_available_conditions():
    """Get list of all available medical conditions for filtering"""
    if not medical_analyzer:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    return {"conditions": medical_analyzer.condition_columns}

@api_router.get("/seasonal-analysis")
async def get_seasonal_analysis():
    """Get detailed seasonal analysis of conditions"""
    if not medical_analyzer:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    try:
        # Analyze different seasons
        seasonal_data = {}
        base_date = datetime(2024, 1, 1)  # Use a reference year
        
        seasons = {
            'Winter': datetime(2024, 1, 15),
            'Spring': datetime(2024, 4, 15),
            'Summer': datetime(2024, 7, 15),
            'Fall': datetime(2024, 10, 15)
        }
        
        for season_name, season_date in seasons.items():
            predictions = medical_analyzer.predict_surges(
                season_date.strftime('%Y-%m-%d'), 
                days_ahead=14
            )
            
            # Calculate average risk for this season
            condition_avgs = {}
            for date, preds in predictions.items():
                for condition, prob in preds.items():
                    if condition not in condition_avgs:
                        condition_avgs[condition] = []
                    condition_avgs[condition].append(prob)
            
            # Get top 10 conditions for this season
            season_summary = {}
            for condition, probs in condition_avgs.items():
                season_summary[condition] = sum(probs) / len(probs)
            
            seasonal_data[season_name] = dict(
                sorted(season_summary.items(), key=lambda x: x[1], reverse=True)[:10]
            )
        
        return {"seasonal_analysis": seasonal_data}
        
    except Exception as e:
        logger.error(f"Seasonal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Seasonal analysis failed: {str(e)}")

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    if not initialize_model():
        logger.error("Failed to initialize medical model on startup")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
