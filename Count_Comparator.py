# -*- coding: utf-8 -*-
"""
Created on Thu May  8 19:23:06 2025

@author: musta
"""

import os

# Klasör yolu (Burayı kendi klasör yolunla değiştir)
folder_path = r"C:\Users\musta\OneDrive\Masaüstü\4.Sınıf Bahar\Bitirme2\Main_Test\Data_Add\dd"

# Dosya adlarını listele
jpeg_files = set()
txt_files = set()

# Klasördeki tüm dosyaları tarama
for file in os.listdir(folder_path):
    if file.endswith(".jpg") or file.endswith(".jpeg"):
        jpeg_files.add(os.path.splitext(file)[0])  # Dosya adını al (uzantısız)
    elif file.endswith(".txt"):
        txt_files.add(os.path.splitext(file)[0])  # Dosya adını al (uzantısız)

# Farklı isimde olanları bul
missing_txt = jpeg_files - txt_files  # JPEG olup TXT olmayanlar
missing_jpeg = txt_files - jpeg_files  # TXT olup JPEG olmayanlar

# Sonuçları yazdır
if missing_txt:
    print("TXT dosyası olmayan JPEG'ler:")
    for file in missing_txt:
        print(file + ".jpg")

if missing_jpeg:
    print("\nJPEG dosyası olmayan TXT'ler:")
    for file in missing_jpeg:
        print(file + ".txt")

if not missing_txt and not missing_jpeg:
    print("Tüm dosyalar eşleşiyor! 🚀")
