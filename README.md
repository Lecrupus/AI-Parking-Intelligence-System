#  Parking Intelligence System

> AI-powered parking analytics platform for detecting parking-induced congestion, identifying enforcement priorities, and supporting smarter urban traffic management.

---

##  Problem Statement

Illegal parking and spillover parking near commercial zones, metro stations, transit hubs, and event venues often reduce roadway capacity and create severe traffic congestion.

Traditional enforcement methods are largely patrol-based and reactive, making it difficult to identify high-impact violations in real time.

This project leverages data analytics and machine learning to transform parking enforcement from a reactive process into a proactive, data-driven system.

---

##  Objectives

- Detect parking congestion hotspots
- Analyze parking violation patterns
- Prioritize enforcement actions using AI
- Support data-driven decision making
- Improve traffic flow and road utilization
- Provide actionable insights through an interactive dashboard

---

##  Key Features

###  Violation Analytics
- Analyze historical parking violations
- Identify recurring problem areas
- Track violation trends over time

###  AI-Based Priority Scoring
- Assign priority scores to violations
- Rank locations based on congestion impact
- Assist enforcement teams in resource allocation

###  Interactive Dashboard
- Visualize parking hotspots
- Monitor enforcement metrics
- Generate operational insights

###  Data Processing Pipeline
- Automated preprocessing
- Feature engineering
- Model training and evaluation
- Dashboard payload generation

---

##  System Architecture

```text
Parking Violation Data
           │
           ▼
Data Cleaning & Processing
           │
           ▼
Feature Engineering
           │
           ▼
Machine Learning Model
           │
           ▼
Priority Scoring Engine
           │
           ▼
Dashboard Visualization
           │
           ▼
Actionable Enforcement Insights
```

---

##  Project Structure

```text
Flipkart_final/
│
├── config/
│   └── prototype_config.json
│
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── scripts/
│   ├── build_prototype.py
│   └── train_priority_model.py
│
├── src/
│   └── parking_intel/
│       ├── __init__.py
│       └── pipeline.py
│
├── dashboard_payload.json
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Frontend | HTML, CSS, JavaScript |
| Configuration | JSON |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Flipkart_final
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### Activate Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Running the Project

### Train the Model

```bash
python scripts/train_priority_model.py
```

### Generate Dashboard Data

```bash
python scripts/build_prototype.py
```

### Launch Dashboard

Open:

```text
dashboard/index.html
```

in your browser.

---

##  Workflow

1. Collect parking violation records
2. Clean and preprocess data
3. Generate engineered features
4. Train priority prediction model
5. Score violations based on impact
6. Generate dashboard-ready output
7. Visualize insights for decision makers

---

##  Expected Outcomes

- Faster identification of congestion hotspots
- Improved enforcement efficiency
- Better allocation of field resources
- Reduced traffic disruptions caused by illegal parking
- Data-driven urban mobility planning

---

##  Use Cases

###  Traffic Police
- Prioritize high-impact violations
- Optimize patrol deployment

###  Smart City Authorities
- Monitor congestion-prone locations
- Support traffic planning initiatives

###  Municipal Corporations
- Evaluate parking policies
- Improve urban mobility strategies

###  Event Management Teams
- Predict parking pressure around venues
- Plan temporary traffic interventions

---

##  Future Enhancements

- Real-time violation ingestion
- GIS and heatmap integration
- Live CCTV analytics
- Congestion prediction models
- Automated alerts and notifications
- Mobile application for field officers
- Smart parking sensor integration

---
## Deployed at :- https://dashboard-steel-sigma-43.vercel.app/
