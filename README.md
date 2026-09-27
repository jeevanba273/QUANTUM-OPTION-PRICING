# ⚛️ Quantum Option Pricing & Risk Analytics 🛡️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway.app-9cf?style=flat-square)](https://quantum-option-pricing.up.railway.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/jeevanba273/QUANTUM-OPTION-PRICING?style=flat-square&color=brightgreen)](https://github.com/jeevanba273/QUANTUM-OPTION-PRICING/stargazers)
[![Open Issues](https://img.shields.io/github/issues/jeevanba273/QUANTUM-OPTION-PRICING?style=flat-square)](https://github.com/jeevanba273/QUANTUM-OPTION-PRICING/issues)

## 📝 About

**Quantum-Option-Pricing** is an educational Flask web application that compares classical option pricing (Black–Scholes and Monte Carlo) with experimental quantum circuits built in **Qiskit** and run on the **Qiskit Aer** simulator. It prices European calls and puts, computes first- and higher-order Greeks, runs experimental quantum Value-at-Risk / Expected Shortfall estimates, and checks model prices against live market data for US and Indian markets.

> **⚠️ Note on the quantum results**
> The quantum circuits in this project are simplified and experimental. They do not run full Quantum Amplitude Estimation, and the reported quantum prices are calibrated against, or blended with, the Black–Scholes price. Names used in code comments for these circuits describe heuristic, hand-tuned encodings, not established or published methods. Treat the quantum outputs as a learning and demonstration tool, not as evidence of quantum advantage.

## 🚀 Live Demo

> **Try it now →** [**QUANTUM OPTION PRICING APP**](https://quantum-option-pricing.up.railway.app/)

## 🔥 Features

| Area | What it does |
|------|--------------|
| **Pricing** | • Black–Scholes (closed form) <br> • Monte Carlo (10,000 geometric Brownian motion paths, NumPy) <br> • Quantum-circuit estimates on the Qiskit Aer simulator |
| **Greeks** | Δ, Γ, Θ, Vega, ρ and higher-order Speed, Zomma, Color, Vanna, Ultima, computed by finite differences and compared with Black–Scholes Greeks |
| **Risk** | Experimental quantum-circuit VaR and Expected Shortfall, compared with historical and parametric (normal) estimates |
| **Sensitivity** | Volatility surface, Greeks heatmap and an illustrative volatility smile (Plotly) |
| **Circuits** | Circuit diagrams, gate counts and depth, and an illustrative Monte Carlo vs QAE resource comparison |
| **Market data** | Live prices via yfinance for US tickers; Indian indices and stocks (NIFTY, BANKNIFTY, `.NS` tickers) with option chains from NSE |
| **API** | JSON endpoints for pricing, Indian option chains and market comparison (see below) |

## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Python 3.9 · Flask · Gunicorn |
| Quantum | Qiskit · Qiskit Aer · qiskit-algorithms |
| Numerics & data | NumPy · SciPy · Pandas · yfinance |
| Charts | Plotly · Matplotlib |
| Frontend | HTML · Bootstrap 5 · Plotly.js |
| Deployment | Railway (Procfile) |

## 🗂️ Project Structure

```text
app/
├── __init__.py                     # Flask app setup
├── routes.py                       # Pages and API endpoints
├── models/
│   ├── classical_models.py         # Black–Scholes and Monte Carlo
│   ├── quantum_models.py           # Demonstration quantum pricing circuits
│   ├── advanced_quantum_models.py  # Experimental circuits, incl. the payoff circuit
│   ├── quantum_greeks.py           # Greeks by finite differences
│   ├── quantum_risk_management.py  # Experimental quantum VaR / Expected Shortfall
│   ├── sensitivity_analyzer.py     # Volatility surface and Greeks heatmap
│   ├── performance_analyzer.py     # Method benchmarks
│   └── validation.py               # Market data and model validation
├── templates/                      # index.html, results.html
└── static/                         # CSS and JavaScript
config.py · run.py · requirements.txt · Procfile · runtime.txt
```

## 📦 Installation (Local Dev)

### 1. Clone
```bash
git clone https://github.com/jeevanba273/QUANTUM-OPTION-PRICING.git
cd QUANTUM-OPTION-PRICING
```

### 2. Create a virtual environment (Python 3.9 recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
python run.py
```
Then open <http://localhost:8080>. The port can be changed with the `PORT` environment variable.

## 🔌 API

| Method | Endpoint | Input |
|--------|----------|-------|
| `POST` | `/api/calculate` | JSON: `stock_price`, `strike_price`, `time_to_maturity`, `risk_free_rate`, `volatility`, `option_type` (`call`/`put`), `num_qubits`, optional `market` (`global`/`india`) and `ticker` |
| `GET` | `/api/indian_options?symbol=NIFTY` | Query parameter `symbol` |
| `POST` | `/compare_market` | Form fields: `ticker`, `market`, `num_options` |

Example:
```bash
curl -X POST http://localhost:8080/api/calculate \
  -H "Content-Type: application/json" \
  -d '{"stock_price": 100, "strike_price": 110, "time_to_maturity": 1, "risk_free_rate": 0.05, "volatility": 0.2, "option_type": "call", "num_qubits": 5}'
```
Returns the Black–Scholes, Monte Carlo and quantum-circuit prices.

## 📚 References

- N. Stamatopoulos, D. J. Egger, Y. Sun, C. Zoufal, R. Iten, N. Shen and S. Woerner, “Option Pricing using Quantum Computers”, *Quantum* 4, 291 (2020).
- Qiskit Finance tutorial, “Pricing European Call Options” (© IBM, Apache License 2.0). The payoff circuit (`_log_normal_circuit` in `app/models/advanced_quantum_models.py`) is modelled on this tutorial.

## 🤖 Development Note

This project was developed with the help of Claude (Anthropic) as an AI coding assistant.

## 📄 License

Released under the [MIT License](LICENSE). © 2025 Jeevan B A.
