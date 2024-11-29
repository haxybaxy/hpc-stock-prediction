# 📈 **Stock Price Prediction with LSTM & GPU Acceleration** 📉

## 🎥 **Video Explanation**
[![Watch the Video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

---

## 📝 **Table of Contents**
1. [🔍 Introduction](#-introduction)
2. [🚀 Features](#-features)
3. [💻 Installation](#-installation)
4. [🛠️ Usage](#️-usage)
5. [📊 Results](#-results)
6. [👥 Multi-Worker Strategy](#-multi-worker-strategy)
7. [⏱️ Profiling & Performance](#️-profiling--performance)
8. [🔒 Security](#-security)
9. [📄 License](#-license)
10. [🙏 Acknowledgements](#-acknowledgements)

---

## 🔍 **Introduction**

Welcome to the **Stock Price Prediction** HPC Final project! 📈✨ This repository contains a Jupyter Notebook (`.ipynb`) that demonstrates how to predict stock prices using a **Long Short-Term Memory (LSTM)** neural network, enhanced with **GPU acceleration** via PyCUDA and a **multi-worker synchronization strategy** leveraging AWS S3. Whether you're a data enthusiast, a machine learning practitioner, or a researcher, this project provides a robust framework for financial forecasting.

---

## 🚀 **Features**

- **Data Acquisition & Visualization** 📊: Fetches historical stock data using `yfinance` and visualizes trends with moving averages.
- **GPU-Accelerated Preprocessing** 🖥️⚡: Utilizes PyCUDA to perform Min-Max Scaling, speeding up data preprocessing.
- **Deep LSTM Architecture** 🧠🔗: Implements a multi-layer LSTM model with dropout regularization to capture complex temporal dependencies.
- **Distributed Training** 👥🌐: Employs a multi-worker strategy with AWS S3 for model synchronization and weight averaging.
- **Performance Profiling** ⏱️📉: Measures training time and evaluates the impact of parallelization on efficiency.
- **Custom CUDA Kernels** 🛠️🔥: Integrates custom CUDA kernels for data scaling and Mean Squared Error (MSE) computation, enhancing computational performance.

---

## 💻 **Installation**

To get started, ensure you have the following prerequisites:

- **Python 3.7+**
- **Jupyter Notebook**
- **CUDA Toolkit** (for GPU acceleration)
- **AWS Account** with access to S3

### 📥 **Clone the Repository**

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
