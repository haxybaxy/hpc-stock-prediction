# 📈 **The Convergence of Machine Learning and HPC in Stock Market Analysis: LSTM RNNs, GPU Acceleration, and Multi-worker Strategies
** 📉

## 🎥 **Video Explanation**
[![Watch the Video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID)

---

## 📝 **Table of Contents**
1. [🔍 Introduction](#-introduction)
2. [🚀 Features](#-features)
3. [📦 Install Required Packages](#-install-required-packages)
4. [🛠️ Usage](#️-usage)
5. [📊 Results](#-results)
6. [👥 Multi-Worker Strategy](#-multi-worker-strategy)
7. [⏱️ Profiling & Performance](#️-profiling--performance)
8. [🔒 Security](#-security)
9. [📄 License](#-license)
10. [🙏 Acknowledgements](#-acknowledgements)

---

## 🔍 **Introduction**

Welcome to our **Stock Price Prediction** HPC final project! 📈✨ This repository contains a Jupyter Notebook (`.ipynb`) that demonstrates how to predict stock prices using a **Long Short-Term Memory (LSTM)** neural network, enhanced with **GPU acceleration** via PyCUDA and a **multi-worker synchronization strategy** leveraging AWS S3. 

---

## 🚀 **Features**

- **Data Acquisition & Visualization** 📊: Fetches historical stock data using `yfinance` and visualizes trends with moving averages.
- **GPU-Accelerated Preprocessing** 🖥️⚡: Utilizes PyCUDA to perform Min-Max Scaling, speeding up data preprocessing.
- **Deep LSTM Architecture** 🧠🔗: Implements a multi-layer LSTM model with dropout regularization to capture complex temporal dependencies.
- **Distributed Training** 👥🌐: Employs a multi-worker strategy with AWS S3 for model synchronization and weight averaging.
- **Performance Profiling** ⏱️📉: Measures training time and evaluates the impact of parallelization on efficiency.
- **Custom CUDA Kernels** 🛠️🔥: Integrates custom CUDA kernels for data scaling and Mean Squared Error (MSE) computation, enhancing computational performance.

---

## 📦 **Install Required Packages**

Open your terminal and install the necessary Python packages:

```bash
pip install boto3
pip install pycuda
pip install pandas
pip install numpy
pip install matplotlib
pip install yfinance
pip install tensorflow
```

---

## 🛠️ **Usage**

### 🔐 **Configure AWS Credentials**

Ensure your AWS credentials are securely set up. Avoid hardcoding them in the notebook. Instead, use environment variables or AWS IAM roles.

```python
import os

os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_ACCESS_KEY'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_KEY'
```

### 📓 **Open the Jupyter Notebook**

Launch Jupyter Notebook and open `HPCprojectWorker2_FINAL_with_scaling.ipynb`.

```bash
jupyter notebook
```

### ▶️ **Run the Notebook**

Execute the cells sequentially. The notebook covers:

- Data downloading and visualization
- GPU-accelerated data preprocessing
- LSTM model definition and training
- Distributed training with multi-worker synchronization
- Performance profiling and results visualization

---

## 📊 **Results**

### 🕒 **Training Time Comparison**

- **Without Parallelization:**
  - **50 Epochs**: ~7 minutes
  - **100 Epochs**: ~40 minutes

- **With Parallelization:**
  - **50 Epochs**: ~4 minutes
  - **100 Epochs**: ~16 minutes

*Parallelization significantly reduces training time, enhancing efficiency and scalability.*

---

### 👥 **Multi-Worker Strategy**

#### Single Worker vs. Five Workers:

The multi-worker approach accelerates training by distributing the workload across multiple instances. While a single worker takes longer to train, five workers collaboratively reduce the overall training time without compromising the model's predictive accuracy.

---

## 👥 **Multi-Worker Strategy**

Our approach employs a **multi-worker synchronization strategy** to enhance training efficiency and model performance. By leveraging AWS S3 as a centralized storage for model checkpoints, multiple workers can save their model states at defined intervals (e.g., every 5 epochs). These models are then retrieved, and their weights are averaged to create a consensus model. This method ensures consistency across workers, reduces training time, and improves the model's generalization by incorporating diverse training signals.

---

## ⏱️ **Profiling & Performance**

Performance profiling was conducted to evaluate the impact of parallelization and multi-worker strategies:

- **Training Time Reduction**: Parallelization cuts down the training duration by over 50%, enabling faster experimentation and iteration.
- **Scalability**: The multi-worker setup scales seamlessly with additional resources, maintaining efficiency even as the number of workers increases.
- **Accuracy**: Despite the reduced training time, the model retains high predictive accuracy, demonstrating the effectiveness of distributed training.

---

## 🔒 **Security**

**Important:** Ensure that AWS credentials are managed securely. Avoid hardcoding sensitive information in the codebase. Instead, use environment variables, AWS IAM roles, or AWS Secrets Manager to handle credentials safely.

```python
import os

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
```

---

We hope this project offers a nice conclusion for our rewarding experience during the High Performance Computing course! 🌟
