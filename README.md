# siamese_csi

# Siamese Neural Networks for Transferable CSI-based Wi-Fi Sensing

This repository contains the core implementation of my Master's thesis on **Channel State Information (CSI)-based human activity recognition** using **Siamese Neural Networks** for improved cross-antenna generalization.

## Overview

WiFi sensing leverages Channel State Information (CSI) to detect and classify human activities without requiring wearable devices. This work addresses a key challenge in practical deployment: **robustness across different WiFi antennas**. Models trained on data from one antenna often fail to generalize to new antennas due to hardware-specific variations in CSI measurements.

The proposed solution uses **Siamese Neural Networks** with metric learning to learn antenna-invariant activity representations, enabling better generalization compared to traditional single-antenna baselines.

## Repository Structure
- single_antenna.py : traditional single-antenna classification baseline
- siamese_model.py : Siamese Neural Network with few shot learning approach
- README.md 


## Files Description

### `single_antenna.py`

Implements a conventional deep learning approach for CSI-based activity recognition:
- Standard classification architecture trained on single-antenna CSI data
- Serves as the baseline for comparison
- Typical performance degradation when tested on unseen antennas

### `siamese_model.py`

Implements the proposed Siamese Neural Network approach:
- Dual-branch architecture processing CSI sample pairs
- **Hybrid loss function**: combination of **contrastive loss** and **classification loss (cross-entropy)**
- Jointly optimizes for both representation similarity and classification accuracy
- Learns similarity metrics that generalize across antennas
- Enables few-shot adaptation to new antennas
## Key Concepts

### Channel State Information (CSI)
CSI captures fine-grained wireless channel properties at the subcarrier level, providing rich information about environmental changes caused by human movement.

### Siamese Neural Networks
A twin-network architecture that processes pairs of inputs through shared weights, learning to map similar activities close together and dissimilar activities far apart in embedding space.

### Hybrid Loss Function
The proposed approach combines:
- **Contrastive Loss**: Pulls embeddings of same-activity pairs closer and pushes different-activity pairs apart
- **Classification Loss (Cross-Entropy)**: Directly optimizes prediction accuracy for activity labels

This multi-objective optimization balances representation learning with discriminative power

### Cross-Antenna Generalization
The ability of a model trained on data from one antenna to maintain performance when deployed on different antennas—a critical requirement for real-world WiFi sensing systems.

## Methodology

1. **Data Preprocessing**: CSI amplitude extraction and normalization
2. **Architecture**: Twin branches with shared parameters + classification head
3. **Loss Function**: Weighted combination of contrastive loss and cross-entropy loss
4. **Training**: Pair sampling with joint optimization of both loss terms
5. **Evaluation**: Cross-antenna testing protocols to measure generalization

## Thesis Context

This implementation supports the experimental results presented in my Master's thesis: **"Siamese Neural Networks for Transferable CSI-based Wi-Fi Sensing"**. The work demonstrates that metric learning approaches significantly improve robustness when deploying WiFi sensing systems across different configurations.

Key findings:
- Siamese networks with hybrid loss outperform single-antenna baselines in cross-antenna scenarios
- The combination of contrastive and classification losses enables both antenna-invariant representations and high accuracy
- The approach supports few-shot adaptation to new antennas with minimal retraining

## Future Directions

Potential extensions of this work include:
- **Domain adaptation** techniques for further robustness
- **Data augmentation** strategies specific to CSI variations
- Multi-antenna fusion approaches
- Real-time deployment on edge devices

