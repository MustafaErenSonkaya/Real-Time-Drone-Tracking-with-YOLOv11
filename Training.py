# -*- coding: utf-8 -*-
"""
Created on Thu May  8 19:53:44 2025

@author: musta
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
    
model = YOLO("yolo11n.pt")

results = model.train(data=r"C:\Users\musta\OneDrive\Masaüstü\4.Sınıf Bahar\Bitirme2\Main_Test\Codes\config.yaml", epochs=80)
