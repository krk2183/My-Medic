import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Calendar, 
  Download, 
  Filter,
  Heart,
  Users,
  TrendingUp,
  Clock,
  Shield,
  BarChart3,
  LineChart,
  PieChart,
  Stethoscope
} from 'lucide-react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
} from 'chart.js';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Alert, AlertDescription } from './ui/alert';
import axios from 'axios';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  Filler
);

const MedicalDashboard = () => {
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState(new Date());
  const [predictionDays, setPredictionDays] = useState(7);
  const [predictions, setPredictions] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [seasonalData, setSeasonalData] = useState(null);
  const [availableConditions, setAvailableConditions] = useState([]);
  const [selectedConditions, setSelectedConditions] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Fetch data on component mount and when date/days change
  useEffect(() => {
    fetchAllData();
  }, [selectedDate, predictionDays]);

  const fetchAllData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        fetchPredictions(),
        fetchMetrics(),
        fetchSeasonalData(),
        fetchAvailableConditions()
      ]);
    } catch (error) {
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPredictions = async () => {
    try {
      const response = await axios.post(`${backendUrl}/api/predict-surge`, {
        start_date: selectedDate.toISOString().split('T')[0],
        days_ahead: predictionDays
      });
      setPredictions(response.data);
    } catch (error) {
      console.error('Error fetching predictions:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${backendUrl}/api/metrics`);
      setMetrics(response.data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchSeasonalData = async () => {
    try {
      const response = await axios.get(`${backendUrl}/api/seasonal-analysis`);
      setSeasonalData(response.data);
    } catch (error) {
      console.error('Error fetching seasonal data:', error);
    }
  };

  const fetchAvailableConditions = async () => {
    try {
      const response = await axios.get(`${backendUrl}/api/available-conditions`);
      setAvailableConditions(response.data.conditions);
      setSelectedConditions(response.data.conditions.slice(0, 10)); // Select top 10 by default
    } catch (error) {
      console.error('Error fetching conditions:', error);
    }
  };

  // Chart configurations
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            family: 'Inter',
            size: 12
          },
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: '#0ea5e9',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 11
          }
        }
      },
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          font: {
            family: 'Inter',
            size: 11
          }
        }
      }
    }
  };

  // Generate weekly trend chart data
  const weeklyTrendData = useMemo(() => {
    if (!predictions) return null;

    const dates = Object.keys(predictions.predictions);
    const topConditions = predictions.seasonal_insights.top_conditions.slice(0, 5);
    
    const datasets = topConditions.map((condition, index) => {
      const colors = [
        'rgba(14, 165, 233, 0.8)',
        'rgba(6, 182, 212, 0.8)',
        'rgba(16, 185, 129, 0.8)',
        'rgba(245, 158, 11, 0.8)',
        'rgba(239, 68, 68, 0.8)'
      ];
      
      return {
        label: condition[0].substring(0, 20) + (condition[0].length > 20 ? '...' : ''),
        data: dates.map(date => predictions.predictions[date][condition[0]] || 0),
        borderColor: colors[index],
        backgroundColor: colors[index].replace('0.8', '0.1'),
        borderWidth: 3,
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointBackgroundColor: colors[index],
        pointBorderColor: '#fff',
        pointBorderWidth: 2
      };
    });

    return {
      labels: dates.map(date => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })),
      datasets
    };
  }, [predictions]);

  // Generate top conditions bar chart data
  const topConditionsData = useMemo(() => {
    if (!predictions) return null;

    const topConditions = predictions.seasonal_insights.top_conditions.slice(0, 10);
    
    return {
      labels: topConditions.map(condition => 
        condition[0].substring(0, 15) + (condition[0].length > 15 ? '...' : '')
      ),
      datasets: [{
        label: 'Average Risk Probability',
        data: topConditions.map(condition => condition[1]),
        backgroundColor: [
          'rgba(14, 165, 233, 0.8)',
          'rgba(6, 182, 212, 0.8)',
          'rgba(16, 185, 129, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(239, 68, 68, 0.8)',
          'rgba(139, 69, 19, 0.8)',
          'rgba(75, 0, 130, 0.8)',
          'rgba(255, 20, 147, 0.8)',
          'rgba(0, 100, 0, 0.8)',
          'rgba(255, 69, 0, 0.8)'
        ],
        borderColor: [
          'rgba(14, 165, 233, 1)',
          'rgba(6, 182, 212, 1)',
          'rgba(16, 185, 129, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(239, 68, 68, 1)',
          'rgba(139, 69, 19, 1)',
          'rgba(75, 0, 130, 1)',
          'rgba(255, 20, 147, 1)',
          'rgba(0, 100, 0, 1)',
          'rgba(255, 69, 0, 1)'
        ],
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false
      }]
    };
  }, [predictions]);

  // Generate seasonal comparison data
  const seasonalComparisonData = useMemo(() => {
    if (!seasonalData) return null;

    const seasons = Object.keys(seasonalData.seasonal_analysis);
    const conditions = Object.keys(seasonalData.seasonal_analysis[seasons[0]] || {}).slice(0, 6);
    
    const datasets = conditions.map((condition, index) => {
      const colors = [
        '#0ea5e9', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'
      ];
      
      return {
        label: condition.substring(0, 15) + (condition.length > 15 ? '...' : ''),
        data: seasons.map(season => seasonalData.seasonal_analysis[season][condition] || 0),
        backgroundColor: colors[index],
        borderColor: colors[index],
        borderWidth: 2,
        borderRadius: 6
      };
    });

    return {
      labels: seasons,
      datasets
    };
  }, [seasonalData]);

  // Risk level distribution data
  const riskDistributionData = useMemo(() => {
    if (!predictions) return null;

    let highRisk = 0, mediumRisk = 0, lowRisk = 0;
    
    Object.values(predictions.predictions).forEach(dayPredictions => {
      Object.values(dayPredictions).forEach(probability => {
        if (probability > 0.3) highRisk++;
        else if (probability > 0.15) mediumRisk++;
        else lowRisk++;
      });
    });

    return {
      labels: ['High Risk (>30%)', 'Medium Risk (15-30%)', 'Low Risk (<15%)'],
      datasets: [{
        data: [highRisk, mediumRisk, lowRisk],
        backgroundColor: [
          'rgba(239, 68, 68, 0.8)',
          'rgba(245, 158, 11, 0.8)',
          'rgba(34, 197, 94, 0.8)'
        ],
        borderColor: [
          'rgba(239, 68, 68, 1)',
          'rgba(245, 158, 11, 1)',
          'rgba(34, 197, 94, 1)'
        ],
        borderWidth: 2
      }]
    };
  }, [predictions]);

  const exportData = () => {
    if (!predictions) return;
    
    const dataToExport = {
      exportDate: new Date().toISOString(),
      predictionPeriod: {
        startDate: selectedDate.toISOString().split('T')[0],
        days: predictionDays
      },
      predictions: predictions.predictions,
      summary: predictions.summary,
      seasonalInsights: predictions.seasonal_insights,
      metrics: metrics
    };
    
    const blob = new Blob([JSON.stringify(dataToExport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medical-surge-predictions-${selectedDate.toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getRiskLevel = (probability) => {
    if (probability > 0.3) return { level: 'High', color: 'bg-red-500', textColor: 'text-red-600' };
    if (probability > 0.15) return { level: 'Medium', color: 'bg-yellow-500', textColor: 'text-yellow-600' };
    return { level: 'Low', color: 'bg-green-500', textColor: 'text-green-600' };
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <div className="loading-spinner mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-slate-700 mb-2">Loading Medical Dashboard</h2>
          <p className="text-slate-500">Analyzing surge predictions...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="bg-white shadow-lg border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-blue-500 to-cyan-500 p-3 rounded-xl">
                <Stethoscope className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-slate-800">Medical Surge Predictor</h1>
                <p className="text-slate-600 mt-1">AI-Powered Healthcare Analytics Dashboard</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button onClick={exportData} className="export-btn">
                <Download className="h-4 w-4" />
                Export Data
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Controls */}
        <div className="bg-white rounded-xl shadow-lg p-6 mb-8 glass">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                <Calendar className="inline h-4 w-4 mr-2" />
                Prediction Start Date
              </label>
              <DatePicker
                selected={selectedDate}
                onChange={setSelectedDate}
                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                dateFormat="yyyy-MM-dd"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                <Clock className="inline h-4 w-4 mr-2" />
                Prediction Days
              </label>
              <Select value={predictionDays.toString()} onValueChange={(value) => setPredictionDays(parseInt(value))}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="7">7 Days</SelectItem>
                  <SelectItem value="14">14 Days</SelectItem>
                  <SelectItem value="30">30 Days</SelectItem>
                  <SelectItem value="60">60 Days</SelectItem>
                  <SelectItem value="90">90 Days</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-2">
                <Activity className="inline h-4 w-4 mr-2" />
                Current Season
              </label>
              <div className="p-3 bg-slate-50 rounded-lg">
                <span className="font-semibold text-slate-800">
                  {predictions?.seasonal_insights?.season || 'Loading...'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Cards */}
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <Card className="metric-card hover-scale bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-blue-600 text-sm font-medium mb-1">Total Conditions</p>
                    <p className="text-3xl font-bold text-blue-800">{metrics.total_conditions}</p>
                  </div>
                  <Users className="h-10 w-10 text-blue-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card hover-scale bg-gradient-to-br from-red-50 to-red-100 border-red-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-red-600 text-sm font-medium mb-1">High Risk Conditions</p>
                    <p className="text-3xl font-bold text-red-800">{metrics.high_risk_conditions}</p>
                  </div>
                  <AlertTriangle className="h-10 w-10 text-red-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card hover-scale bg-gradient-to-br from-green-50 to-green-100 border-green-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-green-600 text-sm font-medium mb-1">Average Risk Score</p>
                    <p className="text-3xl font-bold text-green-800">{(metrics.avg_risk_score * 100).toFixed(1)}%</p>
                  </div>
                  <TrendingUp className="h-10 w-10 text-green-500" />
                </div>
              </CardContent>
            </Card>

            <Card className="metric-card hover-scale bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-purple-600 text-sm font-medium mb-1">Peak Risk Date</p>
                    <p className="text-lg font-bold text-purple-800">
                      {new Date(metrics.peak_risk_date).toLocaleDateString()}
                    </p>
                  </div>
                  <Shield className="h-10 w-10 text-purple-500" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Dashboard Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-white p-2 rounded-xl shadow-lg">
            <TabsTrigger value="overview" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>
            <TabsTrigger value="trends" className="flex items-center space-x-2">
              <LineChart className="h-4 w-4" />
              <span>Trends</span>
            </TabsTrigger>
            <TabsTrigger value="seasonal" className="flex items-center space-x-2">
              <PieChart className="h-4 w-4" />
              <span>Seasonal</span>
            </TabsTrigger>
            <TabsTrigger value="alerts" className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4" />
              <span>Alerts</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Weekly Trend Chart */}
              <Card className="slide-up">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <LineChart className="h-5 w-5 text-blue-500" />
                    <span>Weekly Surge Trends</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="chart-container">
                    {weeklyTrendData && (
                      <Line data={weeklyTrendData} options={chartOptions} />
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Top Conditions Chart */}
              <Card className="slide-up">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BarChart3 className="h-5 w-5 text-green-500" />
                    <span>Top Risk Conditions</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="chart-container">
                    {topConditionsData && (
                      <Bar data={topConditionsData} options={chartOptions} />
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Risk Distribution */}
            <Card className="slide-up">
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <PieChart className="h-5 w-5 text-purple-500" />
                  <span>Risk Level Distribution</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-center">
                  <div style={{ width: '400px', height: '400px' }}>
                    {riskDistributionData && (
                      <Doughnut 
                        data={riskDistributionData} 
                        options={{
                          ...chartOptions,
                          plugins: {
                            ...chartOptions.plugins,
                            legend: {
                              position: 'bottom'
                            }
                          }
                        }} 
                      />
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trends" className="space-y-6">
            <Card className="slide-up">
              <CardHeader>
                <CardTitle>Detailed Trend Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="chart-container" style={{ height: '500px' }}>
                  {weeklyTrendData && (
                    <Line 
                      data={weeklyTrendData} 
                      options={{
                        ...chartOptions,
                        scales: {
                          ...chartOptions.scales,
                          y: {
                            ...chartOptions.scales.y,
                            title: {
                              display: true,
                              text: 'Surge Probability'
                            }
                          },
                          x: {
                            ...chartOptions.scales.x,
                            title: {
                              display: true,
                              text: 'Date'
                            }
                          }
                        }
                      }} 
                    />
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="seasonal" className="space-y-6">
            <Card className="slide-up">
              <CardHeader>
                <CardTitle>Seasonal Comparison Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="chart-container" style={{ height: '400px' }}>
                  {seasonalComparisonData && (
                    <Bar data={seasonalComparisonData} options={chartOptions} />
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="alerts" className="space-y-6">
            {predictions && (
              <div className="space-y-4">
                <h3 className="text-xl font-semibold text-slate-800 mb-4">High-Risk Period Alerts</h3>
                {Object.entries(predictions.high_risk_periods).length > 0 ? (
                  Object.entries(predictions.high_risk_periods).map(([date, conditions]) => (
                    <Alert key={date} className="alert-high">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="font-semibold mb-2">High Risk Alert - {new Date(date).toLocaleDateString()}</div>
                        <div className="space-y-1">
                          {conditions.map((condition, index) => {
                            const risk = getRiskLevel(condition.probability);
                            return (
                              <div key={index} className="flex items-center justify-between">
                                <span className="font-medium">{condition.condition}</span>
                                <Badge className={`${risk.textColor} bg-opacity-20`}>
                                  {(condition.probability * 100).toFixed(1)}% risk
                                </Badge>
                              </div>
                            );
                          })}
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))
                ) : (
                  <Alert className="alert-low">
                    <Shield className="h-4 w-4" />
                    <AlertDescription>
                      <div className="font-semibold">No High-Risk Periods Detected</div>
                      <p>All conditions are within normal risk parameters for the selected period.</p>
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default MedicalDashboard;