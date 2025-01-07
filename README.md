# Mining The Mines

Detecting and classifying mining sites using Sentinel-2 satellite imagery through data mining and machine learning.

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Goals and Impact](#goals-and-impact)
- [Data and Methodology](#data-and-methodology)
  - [Dataset](#dataset)
  - [Techniques](#techniques)
- [Key Features](#key-features)


---

## Introduction

This repository contains the resources for a project aimed at detecting mining sites using Sentinel-2 satellite imagery. It addresses the pressing need for better monitoring of mineral resource extraction, particularly in regions lacking sufficient resources to regulate mining activities.

---

## Problem Statement

Mineral resources are vital for daily life and technological advancements. However, many regions, particularly in Africa and Southeast Asia, face challenges such as:

- Inability to monitor and regulate mining activities effectively.
- Environmental degradation and health risks due to poor mining practices.
- Economic and social challenges related to unauthorized mining.

This project seeks to tackle these challenges using machine learning and satellite imagery.

---

## Goals and Impact

1. **Environmental Conservation**: Prevent unauthorized mining activities.
2. **Economic Benefits**: Support sustainable local economies.
3. **Social Harmony**: Reduce conflicts and exploitation associated with mining.
4. **Health Improvements**: Minimize health risks caused by unsafe mining practices.

---

## Data and Methodology

### Dataset

The project utilizes Sentinel-2 multispectral data, provided by the European Space Agency (ESA). Key features include:

- **Image Details**: 12 spectral bands, each with 512x512 resolution.
- **Training Dataset**: 1242 images (20% positive samples).
- **Evaluation Dataset**: 1243 images.
- **Preprocessing**: Includes cloud masking and generating median images for specific locations.

### Techniques

- **Preprocessing**: Normalization, spectral indices, feature extraction, and augmentation.
- **Classification Model**:
  - MaxViT (Vision Transformer) leveraging both convolutional and transformer architectures.
  - Binary segmentation for detailed mapping of mining regions.
- **Evaluation Metrics**:
  - F1 score, loss metrics, and IoU for validation.

---

## Key Features

1. **Modeling**:
   - MaxViT for robust image classification.
   - CNNs for binary segmentation tasks.

2. **Performance**:
   - High classification accuracy with well-documented evaluation metrics.

3. **Potential Applications**:
   - Mapping mining sites.
   - Informing policy decisions for sustainable mining practices.


