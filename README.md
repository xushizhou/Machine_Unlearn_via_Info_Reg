# Machine Unlearning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Overview

This repository serves as the **official hub** for research and development on **Machine Unlearning via Information-Theoretic Regularization**, an approach designed to **remove the marginal effect of specific features or data points while preserving model utility**. The proposed framework is **method-agnostic** and provides a unified perspective on unlearning through **optimal transport and information-theoretic constraints**.

We welcome **collaborations, discussions, and contributions** to advance this framework and explore its integration into open-source libraries.

### Why This Framework?
- **A Unified Theoretical Perspective on Model Unlearning**: Built on **information-theoretic regularization**, ensuring rigorous unlearning guarantees while maintaining utility.
- **Multipurpose Direct Feature Unlearning**: Enables direct feature unlearning **at the data level** while maximizing retained information for **general downstream tasks**, without requiring predefined target variables.
- **Open-Source Collaboration**: Contributions are encouraged for both theoretical advancements and practical implementations.

---

## 🤝 How to Get Involved

We invite **researchers, developers, and practitioners** to collaborate in the following areas:  

1️⃣ **Implementing Multipurpose Feature Unlearning on Data** using Wasserstein Barycenters.  
2️⃣ **Enhancing Data Point Unlearning** by improving information-theoretic regularization to **better penalize the marginal effect** of adding/removing data points.  
3️⃣ **Exploring Alternative Regularizations** to refine unlearning performance beyond **mutual information**, tailored to specific learning objectives.  
4️⃣ **Benchmarking & Evaluation**: Comparing against existing unlearning approaches (e.g., **amnesic unlearning**) and testing against adversarial attacks (e.g., **membership inference attacks**) on datasets of various scales.  
5️⃣ **Precise Estimation of Retraining from Scratch**: Investigating the correlation between **mutual information and utility** to refine **regularization parameter tuning** and improve **retraining estimation accuracy**.  
6️⃣ **Open-Source Integration**: Expanding the framework into **scikit-learn**, `TorchUnlearn`, and other ML libraries.  

If you're interested in contributing, please **open an issue** or **start a discussion** on GitHub.  
For specific research inquiries, feel free to reach out: **Shizhou Xu**.

---

## The Proposed Framework

The framework formulates **feature unlearning** and **data point unlearning** as **optimization problems** with information-theoretic constraints:

- **Multi-Purpose Feature Unlearning on Data (Optimal Transport-Based)**:
  $$
  \sup_{f: \mathcal{X} \times \mathcal{Z} \rightarrow \mathcal{X}} \mathcal{U}(X; \hat{X}) \quad \text{subject to} \quad \hat{X} \perp Z
  $$
  where $\hat{X}$ is the transformed dataset that retains utility while **removing unwanted feature information**.

- **Data Point Unlearning (Mutual Information-Based)**:
  $$
  \sup_{f: \mathcal{X} \rightarrow \mathcal{Y}} \mathcal{U}(Y; \hat{Y}) - \gamma I(\hat{Y}; Z)
  $$
  where the goal is to balance utility preservation with **data point removal** via **soft independence constraints**.

---

## 📦 Installation (Coming Soon)

The repository will host modular implementations for:
- Wasserstein Barycenters (Feature Unlearning)
- Mutual Information Regularization (Data Point Unlearning)
- Synthetic and Real-World Dataset Experiments

For now, **stay updated** by starring this repository ⭐ and following our progress.
