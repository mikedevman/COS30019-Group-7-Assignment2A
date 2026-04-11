# COS30019 Assignment 2B — Traffic-Based Route Guidance System (TBRGS)

## Project Overview

This system predicts traffic flow using deep learning and uses those predictions to find the top-K fastest routes between any two SCATS intersections in the Boroondara area of Melbourne.

### Machine Learning Models
- **LSTM** — 2-layer Long Short-Term Memory network 
- **Bidirectional LSTM** — LSTM that reads sequences in both forward and backward directions
- **GRU** — 2-layer Gated Recurrent Unit network 
- **Bidirectional GRU** — GRU that reads sequences in both forward and backward directions
- **Custom GCN-LSTM** — A custom-built Graph Convolutional Network combined with LSTM that models both spatial relationships between intersections and temporal traffic patterns simultaneously

### Data Processing
- Raw SCATS data is reshaped from wide format (V00–V95 columns) to long format (one row per 15-minute interval)
- Traffic volumes are aggregated per SCATS site and timestamp
- Engineered features include: time of day, day of week, weekend flag, rush hour flag, and night flag
- For GCN-LSTM: additional cyclical time encodings (sin/cos) and lag features (15 min, 1 hour, 1 day prior)
- A KNN spatial adjacency matrix is built from SCATS site coordinates for graph-based modelling

### Routing
- Predicted traffic volume is converted to travel time using the VicRoads speed-flow formula (quadratic model)
- Assumes a 60 km/h speed limit and a 30-second delay per controlled intersection
- Routes are found using **Yen's K-Shortest Paths** algorithm built on top of **A\***
- Returns up to 5 optimal routes between any origin–destination pair of SCATS sites

### GUI
- React + Leaflet web interface displaying routes on an interactive OpenStreetMap
- Route lines are snapped to real roads using the OSRM routing engine
- Sidebar controls for model selection, departure time, speed limit, intersection delay, and top-K routes
- Route cards showing estimated travel time, distance, and number of intersections per route

---

## How to Run

