import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MedicalDashboard from './components/MedicalDashboard';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<MedicalDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;