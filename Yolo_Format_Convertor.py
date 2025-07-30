# -*- coding: utf-8 -*-
"""
Created on Thu May  8 19:08:30 2025

@author: musta
"""

import os
import xml.etree.ElementTree as ET

# XML dosyalarının bulunduğu klasör
xml_folder = r"C:\Users\musta\OneDrive\Masaüstü\4.Sınıf Bahar\Bitirme2\Drone_Veri_Seti"
output_folder = r"C:\Users\musta\OneDrive\Masaüstü\4.Sınıf Bahar\Bitirme2\YOLO"

# Etiket sınıfları (Her sınıfa bir ID ata)
class_dict = {"drone": 0}  

def convert_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    yolo_annotations = []
    
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name in class_dict:
            class_id = class_dict[class_name]
        else:
            continue  # Bilinmeyen sınıfları atla
        
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        
        # YOLO formatına normalize edilmiş dönüşüm
        x_center = (xmin + xmax) / 2 / img_width
        y_center = (ymin + ymax) / 2 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_annotations

# Tüm XML dosyalarını işle
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(xml_folder):
    if filename.endswith(".xml"):
        xml_path = os.path.join(xml_folder, filename)
        
        # Görüntü boyutlarını belirleme (Eğer tüm görüntüler aynıysa, elle girilebilir)
        img_width, img_height = 512, 512  
        
        yolo_labels = convert_to_yolo(xml_path, img_width, img_height)
        
        # Çıktıyı aynı isimli .txt dosyasına kaydet
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)
        
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_labels))
        
        print(f"{txt_filename} başarıyla oluşturuldu!")

print("Tüm XML dosyaları YOLO formatına dönüştürüldü! 🚀")
