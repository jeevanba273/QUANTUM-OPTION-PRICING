# ⚛️ Quantum-Enhanced Option Pricing & Risk Analytics 🛡️

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway.app-9cf?style=flat-square)](https://quantum-option-pricing.up.railway.app/)
[![Stars](https://img.shields.io/github/stars/jeevanba273/QUANTUM-OPTION-PRICING?style=flat-square&color=brightgreen)](https://github.com/jeevanba273/QUANTUM-OPTION-PRICING/stargazers)
[![Open Issues](https://img.shields.io/github/issues/jeevanba273/QUANTUM-OPTION-PRICING?style=flat-square)](https://github.com/jeevanba273/QUANTUM-OPTION-PRICING/issues)

## 📝 Abstract

**Quantum-Option-Pricing** is a research-grade web application that integrates classical models (Black–Scholes, Monte Carlo) with **Amplitude Estimation**, **Quantum Walk**, and **Variational Quantum** circuits to price European options, compute higher-order Greeks, and perform real-time risk analytics.  

The platform is purpose-built for **Financial Engineering**, showcasing tangible **Quantum Advantage** while remaining production-ready for cloud deployment.

## 🚀 Live Demo

> **Try it now →** [**QUANTUM OPTION PRICING APP**](https://quantum-option-pricing.up.railway.app/)  

## 🔥 Key Features

| Category | Highlights |
|----------|------------|
| **Pricing Engines** | • Black–Scholes (closed-form) <br> • GPU-accelerated Monte Carlo <br> • Quantum Amplitude Estimation (QAE) <br> • Variational-Quantum Eigensolver (VQE) |
| **Risk Metrics** | Live computation of Δ, Γ, Θ, ρ, Vega <br> Higher-order Greeks: Speed, Zomma, Color, Vanna, Ultima |
| **Quantum Tooling** | Dynamic circuit synthesis with Qiskit ℠ <br> Noise-aware simulation (Aer) & IBM Q backend hooks |
| **Scalability** | Async FastAPI backend, Redis task queue, PostgreSQL persistence |
| **Deployment** | Docker-first workflow → CI/CD → Railway.app <br> One-click GitHub deploy |
| **Visual UI** | Tailwind + HTMX for snappy SPA-feel without JS bloat <br> Dark-mode, responsive charts (Chart.js) |


## 🛠️ Tech Stack

| Layer | Tools |
|-------|-------|
| Frontend | HTML 5 · Tailwind CSS · HTMX · Alpine.js |
| Backend | Python 3.9 · Flask · Pydantic · SQLModel |
| Quant Libs | Qiskit Terra/Aer · NumPy · SciPy · Pandas · TA-Lib |
| Deployment | Railway.app · Docker |
| Testing | PyTest · Coverage.py · Hypothesis |

## 📦 Installation (Local Dev)

### 1. Clone
```bash
git clone https://github.com/jeevanba273/QUANTUM-OPTION-PRICING.git
cd QUANTUM-OPTION-PRICING
```
