# 🚗 VANET ML + Game Theory Project

## 📌 Overview
This project implements an intelligent VANET system using:
- Machine Learning (Cluster Head Selection)
- Game Theory (Dynamic Optimization)
- ns-3 (Network Simulation)
- SUMO (Mobility)
- NetAnim (Visualization)
- Streamlit (Dashboard)

---

## ⚙️ Pipeline
SUMO → ML → Game Theory → ns-3 → FlowMonitor → NetAnim

---

## 📊 Results
| Method | PDR (%) | Delay (s) | Throughput |
|-------|--------|----------|-----------|
| Static ML | 79.37 | 0.0069 | 0.006 |
| Dynamic GT | 77.88 | 0.0163 | 0.002 |

---

## 📡 Routing Comparison
| Protocol | PDR (%) |
|----------|--------|
| AODV | 23 |
| OLSR | 68 |

---

## 🎥 Visualization
NetAnim is used for packet-level animation.

---

## 🚀 How to Run

```bash
cd ~/VANET-ML-GameTheory
source .venv/bin/activate
streamlit run dashboard.py
