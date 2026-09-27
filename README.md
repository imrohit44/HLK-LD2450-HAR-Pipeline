# HLK-LD2450 Human Activity Recognition (HAR) Pipeline

An end-to-end Machine Learning pipeline for Human Activity Recognition (Walking, Standing, Sitting, Lying, Bending/Squatting, Fall) using the HLK-LD2450 24GHz mmWave Radar Sensor. 

This repository contains everything needed to generate synthetic radar data, train a classification model, and deploy it to both PC (via Python/PySerial) and edge microcontrollers (via C++/Arduino IDE).

##  Features
* **Synthetic Data Generator:** Mathematically models LD2450 UART outputs (X, Y, Speed) to create robust training data without manual logging.
* **Feature Engineering Pipeline:** Extracts sliding-window time-series statistics (mean, variance, deltas, peaks) suitable for tabular ML models.
* **Dual Deployment:**
  * **Python Edge (PC/Raspberry Pi):** Real-time inference script using `pyserial` and `joblib`.
  * **Microcontroller Edge (ESP32/Arduino):** Standalone C++ header file generated via `micromlgen` for direct hardware deployment.

##  Hardware Requirements
* HLK-LD2450 24GHz mmWave Radar Sensor
* USB-to-TTL Serial Adapter (for PC deployment)
* ESP32 or Arduino compatible microcontroller (for edge deployment)
* Jumper wires

