# 🗣️ Speaker Notes: AML Project 4 Presentation
**CSCI-6767 Applied Machine Learning & Data Analytics**
**Presenters:** Gabil Gurbanov & Hamida Hagverdiyeva

## 1. Project Introduction
* **The Goal:** We applied machine learning to two distinct real-world tasks to compare how different model families handle complex data.
* **Task 1 (Regression):** Predicting Seoul Bike Sharing demand.
* **Tasks 2 & 3 (Classification):** Detecting cyber attacks in the KDD Cup 1999 Network Intrusion dataset.
* **Our Approach:** We didn't just chase high accuracy scores. We focused on *why* models succeed, when they fail, and their real-world interpretability.

---

## 2. 🚴‍♂️ Task 1: Non-Linear Regression (Seoul Bike Demand)
* **The Problem with Linear Models:** A straight line assumes rentals go up forever as temperature goes up. In reality, rentals have a "sweet spot" (e.g., 20°C) and drop in extreme heat. Hourly demand also forms a complex "M" shape (peaks at 8 AM and 5 PM). We needed flexible models.
* **The Contenders:**
    * **Polynomial:** Captures the general trend but misses subtle dips.
    * **Step Functions:** Chops data into blocks, but creates unnatural, sudden jumps.
    * **LOESS:** "Overthinks" the data, wiggling too much to follow random noise (overfitting).
    * **Splines:** Smoothly ties flexible curves together.
* 🏆 **The Winner — GAM (Generalized Additive Model):** * *Why we chose it:* It allows us to pick the perfect non-linear shape (like a spline) for each individual feature, then adds them up. It provides the best balance of high accuracy and human readability.
* **Data Insights to Mention:**
    * *Log-Transform:* We log-transformed the target variable because the raw data was heavily skewed (most hours have very few rentals).
    * *Multicollinearity:* Noticed a near-perfect correlation between Temperature and Dew Point. Dew Point is often a better predictor of actual human comfort.

---

## 3. 🛡️ Tasks 2 & 3: Classification (Network Intrusion Detection)
* **The Problem:** Identifying if a network packet is "Normal" or an "Attack".
* **Family 1: Tree Ensembles 🌲 (The Committee)**
    * *Concept:* Multiple decision trees voting on an outcome. Highly interpretable "if/then" rules.
    * *Bagging:* Trains trees on random subsets of data to prevent overfitting.
    * 🏆 **Random Forest (Top Pick):** Upgrades Bagging by also forcing trees to look at *random features*. This stops every tree from just looking at the most obvious feature, making the final vote incredibly robust.
    * *Gradient Boosting:* Builds trees sequentially, where each new tree fixes the mistakes of the last one.
* **Family 2: Support Vector Machines 📏 (The Border Patrol)**
    * *Concept:* Drawing mathematical boundaries between normal and attack data.
    * 🏆 **RBF Kernel (Top SVM):** Draws tight "bubbles" around clusters. Handled the network data perfectly because attacks group in identical clusters.
    * ❌ **Sigmoid Kernel (The Loser):** Forced an S-shaped boundary, cutting through the wrong data and dropping to ~85% accuracy.

---

## 4. 📊 Addressing the Visualizations & Data Limitations
*(Crucial talking points to show critical thinking)*

* **The "Weird" Seasonal Graph:** The original app graph showed equal bars for all seasons. That graph was just counting *rows* in the dataset (hours per season), not actual bike rentals. The true average rentals peak in Summer and Autumn.
* **The 99.9% Accuracy Illusion (KDD99 Dataset):**
    * Our Tree and RBF models achieved near 100% accuracy.
    * *The Catch:* We acknowledge this is a quirk of the 1999 dataset, which contains millions of duplicate attack signatures. The models easily memorized them.
    * *Real-World Fix:* A model trained on 1999 data won't catch 2026 attacks. We recommend testing on the modern **UNSW-NB15** benchmark.
* **The "Zoomed-In" Bar Chart:** Our original comparison chart started the Y-axis at 0.8, exaggerating the failure of the Sigmoid SVM. We corrected this to start at 0.0 for total transparency.

---

## 5. 🎯 Final Takeaways (Closing Slide)
* **For Regression:** We recommend the **GAM** because it significantly outperforms linear baselines while remaining completely interpretable, allowing us to see exactly how features shape demand.
* **For Classification:** We recommend **Random Forest** because it delivers near-perfect accuracy, scales easily, and provides clear "feature importance" rules (unlike the SVM "black box"), so we know exactly why a connection was flagged.

