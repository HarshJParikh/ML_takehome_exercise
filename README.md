# 🚀 Multi-Task Learning for Sentiment Analysis & Intent Detection

## 📌 Project Overview
This project implements a **multi-task learning (MTL) model** for:
1. **Sentiment Analysis** → Predicts whether a sentence is **positive or negative**.
2. **Intent Detection** → Classifies a sentence into one of four intent categories:  
   - **Question**
   - **Statement**
   - **Command**
   - **Exclamation**

The model is based on **all-MiniLM-L6-v2** from `sentence-transformers` and is trained using **synthetic datasets**.

---

## ⚙️ Setup & Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/HarshJParikh/ML_takehome_exercise
cd ml_takehome_exercise
```

### **2️⃣ 🐳 Running with Docker**
#### **Build the Docker Image**
```bash
docker build -t ml-takehome .
```
#### **Run the Docker Container**
```bash
docker run --rm ml-takehome
```

### **📂 Project Structure**
```bash
📁 ml-takehome-project/
│── 📄 Dockerfile                      # Docker setup for the project
│── 📄 requirements.txt                 # Python dependencies
│── 📄 ml_takehome_exercise.py          # Main script for training & inference
│── 📄 Unique_Fixed_Sentiment_Analysis_Dataset.csv   # Sentiment dataset
│── 📄 Unique_Intent_Classification_Dataset.csv      # Intent dataset
│── 📄 README.md                         # Documentation
│── 📄 ML_Apprentice_Take_Home_Assignment    # Project Description
```



