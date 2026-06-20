# Parking Intelligence System

An AI-powered Parking Intelligence and Enforcement Analytics platform designed to identify parking-induced congestion hotspots, prioritize violations, and provide actionable insights through an interactive dashboard.

## Overview

Urban areas often suffer from traffic congestion caused by illegal parking, spillover parking near commercial zones, transit hubs, and event locations. This project combines data analytics, machine learning, and visualization to help authorities:

- Detect parking congestion hotspots
- Prioritize enforcement actions
- Analyze parking violation trends
- Generate operational insights
- Visualize results through a web dashboard

---

## Project Structure


Flipkart_final/
│
├── config/
│ └── prototype_config.json
│
├── dashboard/
│ ├── app.js
│ ├── index.html
│ └── styles.css
│
├── scripts/
│ ├── build_prototype.py
│ └── train_priority_model.py
│
├── src/
│ └── parking_intel/
│ ├── init.py
│ └── pipeline.py
│
├── dashboard_payload.json
├── requirements.txt
└── README.md


---

## Key Features

### Parking Violation Analytics
- Aggregates parking violation records
- Identifies high-frequency violation zones
- Detects recurring congestion patterns

### Priority Scoring Engine
- Assigns priority scores to violations
- Helps enforcement teams focus on critical locations
- Supports data-driven decision making

### Machine Learning Pipeline
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Priority prediction generation

### Interactive Dashboard
- Visual representation of violation trends
- Hotspot identification
- Enforcement recommendations
- Operational monitoring

---

## Technology Stack

### Backend
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn

### Frontend
- HTML
- CSS
- JavaScript

### Configuration
- JSON-based configuration files

---

DEPLOYED AT:-https://dashboard-steel-sigma-43.vercel.app/

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Flipkart_final
2. Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Linux/Mac

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Running the Project
Train the Priority Model
python scripts/train_priority_model.py
Build the Prototype Dataset
python scripts/build_prototype.py
Run the Dashboard

Open:

dashboard/index.html

in your browser.

Data Flow
Input parking violation data
Data cleaning and preprocessing
Feature extraction
Priority model training
Score generation
Dashboard payload creation
Visualization and decision support
Use Cases
Traffic Police
Target high-impact violations
Improve enforcement efficiency
Smart Cities
Monitor congestion hotspots
Support urban mobility planning
Municipal Authorities
Allocate enforcement resources
Evaluate policy effectiveness
Event Management
Predict parking pressure near venues
Plan temporary enforcement measures
Configuration

System parameters can be modified through:

config/prototype_config.json

This file controls:

Model parameters
Threshold values
Processing settings
Dashboard configuration
Future Enhancements
Real-time violation ingestion
GIS map integration
Live camera feeds
Predictive congestion forecasting
Automated alert generation
Mobile enforcement application
Integration with smart parking sensors
Expected Impact

The Parking Intelligence System enables authorities to move from reactive enforcement to proactive, data-driven traffic management by identifying congestion-causing parking behaviors before they significantly affect road operations.



License

This project is intended for educational, research, and prototype development purposes.


#
