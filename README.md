# 💧 Image-Based Turbidity Estimation

This project provides a **Python-based image processing system** that estimates **water turbidity (NTU)** using region-based red-channel analysis. It offers:

* 📌 **Absolute Mode** – Uses an exponential model to directly calculate NTU.
* 🔁 **Relative Mode** – Device-independent method using a 0 NTU reference.

---

## 🧠 Project Objective

To simulate the functionality of commercial turbidity meters using image-based techniques for educational, environmental, and research use, ensuring low cost and high usability.

---

## 🛠️ Features

* Region-based NTU extraction (Top, Center, Bottom)
* Accurate red channel analysis
* Exponential NTU prediction model
* Relative comparison with 0 NTU reference
* UI built with React + TailwindCSS for better user interaction

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/turbidity-estimation.git
cd turbidity-estimation
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Python script

```bash
python main.py
```

### 4. Run the React UI (Frontend folder)

```bash
cd frontend
npm install
npm run dev
```

---

## 💡 UI Design Overview

The frontend is implemented using **React + TailwindCSS** with the following features:

### Modes:

* ✅ Absolute Mode (Single sample image)
* ✅ Relative Mode (Reference + Sample)

### Upload Sections:

* 📤 Upload reference image (only in relative mode)
* 📤 Upload sample image

### Output:

* 📊 Final NTU value display
* 📈 Region-wise NTU summary
* 📥 Report download (JSON)

---

## 📷 User Instructions

### 🧪 Absolute Mode

1. Select **"Absolute Measurement"**
2. Upload a sample image
3. Click **Analyze Turbidity**
4. See final NTU and regional values

### 🔁 Relative Mode

1. Select **"Relative Measurement"**
2. Upload a **0 NTU reference image**
3. Upload the unknown **sample image**
4. Click **Analyze Turbidity**
5. View relative NTU calculated using red channel intensity division

---

## 🔁 Device Independent NTU Estimation

* Uses red channel intensity ratio between sample and reference images
* Applies normalization for better device-independence
* Provides meaningful relative NTU outputs even under varying conditions

---

## 🧰 Common Git Commands

```bash
# Clone repository
git clone https://github.com/your-username/turbidity-estimation.git

# View project status
git status

# Stage and commit changes
git add .
git commit -m "Your message"

# Push changes to GitHub
git push origin main

# Pull latest updates
git pull origin main
```

---

## 📐 UI Development Instructions

### 🛠 Tech Stack

* React
* TailwindCSS
* Lucide Icons
* JavaScript (ES6+)

### 📋 Components

1. **ModeSelector** – Absolute / Relative mode toggle
2. **ImageUploader** – Upload reference and sample images
3. **Analyzer** – Triggers analysis and computes NTU
4. **ResultDisplay** – Shows final NTU and region data
5. **DownloadReport** – JSON download functionality

### 🎨 UI Instructions (Step-by-Step)

1. **Create new React project** using Vite or CRA

```bash
npm create vite@latest turbidity-analyzer --template react
cd turbidity-analyzer
npm install
```

2. **Install TailwindCSS**

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Edit `tailwind.config.js` and `index.css` with Tailwind setup

3. **Create components**: `ModeSelector.jsx`, `ImageUploader.jsx`, `ResultDisplay.jsx`, etc.

4. **Use useRef to manage image inputs** and FileReader to get preview

5. **Simulate image processing with static or backend data** (or use real backend via API call)

6. **Design NTU Output Section**

```jsx
<div className="text-6xl font-bold text-center text-blue-600">{ntu.toFixed(2)} NTU</div>
```

7. **Download button to export results**

```jsx
const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
```

---

## 📦 Output

* **Absolute Mode**: Direct NTU value from 3-region red mean
* **Relative Mode**: Normalized NTU based on reference
* **Downloadable Report**: `.json` containing all results

---

## 📬 Feedback

For suggestions or bug reports, feel free to open an issue or contact the author.

---

**Developed with ❤️ for real-world water monitoring applications**
