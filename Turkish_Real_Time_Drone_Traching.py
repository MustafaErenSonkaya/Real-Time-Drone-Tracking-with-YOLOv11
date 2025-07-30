# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 21:18:58 2025

@author: mustaas
"""

# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import torch
import psutil
import time
import threading

if torch.cuda.is_available():
    #device = 'cuda:0'
    device = torch.device('cuda:0')
    print(f"CUDA kullanilabilir! GPU kullaniliyor: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Surumu: {torch.version.cuda}")
    print(f"GPU Bellegi: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("CUDA kullanilamiyor, CPU kullaniliyor")

model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')
model = YOLO(model_path)  
model.to(device)

esik = 0.3
iou_esigi = 0.5
parlaklik_ayari = 0.0
kontrast_ayari = 1.0

on_isleme_modlari = {
    0: "Hicbiri",
    1: "Histogram Esitleme", 
    2: "CLAHE (Uyarlanabilir)",
    3: "Gaussian Bulaniklastirma + Keskinlestirme",
    4: "Kenar Gelistirme",
    5: "Gurultu Azaltma",
    6: "Gamma Duzeltme",
    7: "Renk Uzayi (HSV)",
    8: "Kontrast Germe",
    9: "Sobel Kenar Tespiti",
    10: "Sobel X-Yonu",
    11: "Sobel Y-Yonu",
    12: "Sobel Birlesik + Orijinal"
}
mevcut_on_isleme = 0

tespit_gecmisi = defaultdict(list)
maks_gecmis = 5

sistem_istatistikleri = {
    'cpu_yuzdesi': 0.0,
    'ram_yuzdesi': 0.0,
    'ram_kullanilan_gb': 0.0,
    'ram_toplam_gb': 0.0,
    'gpu_bellek_yuzdesi': 0.0,
    'gpu_kullanimi': 0.0,
    'gpu_sicakligi': 0.0
}

def sistem_istatistiklerini_guncelle():
    global sistem_istatistikleri
    while True:
        try:
            sistem_istatistikleri['cpu_yuzdesi'] = psutil.cpu_percent(interval=0.1)
            ram = psutil.virtual_memory()
            sistem_istatistikleri['ram_yuzdesi'] = ram.percent
            sistem_istatistikleri['ram_kullanilan_gb'] = ram.used / (1024**3)
            sistem_istatistikleri['ram_toplam_gb'] = ram.total / (1024**3)
            if device == 'cuda:0':
                try:
                    bellek_kullanilan = torch.cuda.memory_allocated(0)
                    bellek_toplam = torch.cuda.get_device_properties(0).total_memory
                    sistem_istatistikleri['gpu_bellek_yuzdesi'] = (bellek_kullanilan / bellek_toplam) * 100
                    sistem_istatistikleri['gpu_kullanimi'] = min(100, sistem_istatistikleri['gpu_bellek_yuzdesi'] * 1.5)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        kulp = pynvml.nvmlDeviceGetHandleByIndex(0)
                        sicaklik = pynvml.nvmlDeviceGetTemperature(kulp, pynvml.NVML_TEMPERATURE_GPU)
                        sistem_istatistikleri['gpu_sicakligi'] = sicaklik
                    except:
                        sistem_istatistikleri['gpu_sicakligi'] = 0.0
                except Exception as e:
                    print(f"GPU istatistikleri alinamadi: {e}")
            time.sleep(0.5)
        except Exception as e:
            print(f"Sistem izlemede hata: {e}")
            time.sleep(1)

kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("Hata: Web kamerasi acilamadi")
    exit()

kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("GPU hizlandirmali gercek zamanli drone tespiti sistem izleme ile baslatiliyor...")
print(f"device: {device.upper()}")
if device == 'cuda:0':
    print("YOLO cikarimi icin GPU hizlandirmasi etkin!")
print("Kontroller:")
print("'q' veya ESC - cik")
print("'o' / 'p' - Parlakligi artir/azalt (±0.1)")
print("'c' / 'v' - Kontrasti artir/azalt (±0.1)")
print("'r' - Parlaklik ve kontrasti sifirla")
print("'f' - Kare yumusatmayi ac/kapa")
print("'t' - Guven esiklerini degistir")
print("'m' - on isleme modlarini degistir (Sobel filtreleri dahil)")
print("'h' - on isleme yardimini goster")
print("'s' - Ayrintili sistem istatistiklerini goster")

kare_yumusatma = True
esik_degerleri = [0.2, 0.3, 0.4, 0.5, 0.6]
esik_indeksi = 1

def sobel_filtresi_uygula(kare, yon='her_ikisi'):
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    bulanik = cv2.GaussianBlur(gri, (3, 3), 0)
    if yon == 'x':
        sobel = cv2.Sobel(bulanik, cv2.CV_64F, 1, 0, ksize=3)
    elif yon == 'y':
        sobel = cv2.Sobel(bulanik, cv2.CV_64F, 0, 1, ksize=3)
    else:
        sobel_x = cv2.Sobel(bulanik, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(bulanik, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def kareyi_on_isle(kare, parlaklik=0.0, kontrast=1.0, mod=0):
    if parlaklik != 0.0 or kontrast != 1.0:
        kare = cv2.convertScaleAbs(kare, alpha=kontrast, beta=parlaklik)
    if mod == 0:
        return kare
    elif mod == 1:
        yuv = cv2.cvtColor(kare, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    elif mod == 2:
        lab = cv2.cvtColor(kare, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif mod == 3:
        bulanik = cv2.GaussianBlur(kare, (3, 3), 0)
        cekirdek = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
        return cv2.filter2D(bulanik, -1, cekirdek)
    elif mod == 4:
        gaussian = cv2.GaussianBlur(kare, (0, 0), 2.0)
        return cv2.addWeighted(kare, 1.5, gaussian, -0.5, 0)
    elif mod == 5:
        return cv2.bilateralFilter(kare, 9, 75, 75)
    elif mod == 6:
        gamma = 1.2
        ters_gamma = 1.0 / gamma
        tablo = np.array([((i / 255.0) ** ters_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(kare, tablo)
    elif mod == 7:
        hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= 1.2
        hsv[:,:,2] *= 1.1
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif mod == 8:
        p2, p98 = np.percentile(kare, (2, 98))
        return cv2.convertScaleAbs(kare, alpha=255.0/(p98-p2), beta=-p2*255.0/(p98-p2))
    elif mod == 9:
        return sobel_filtresi_uygula(kare, 'her_ikisi')
    elif mod == 10:
        return sobel_filtresi_uygula(kare, 'x')
    elif mod == 11:
        return sobel_filtresi_uygula(kare, 'y')
    elif mod == 12:
        sobel_kare = sobel_filtresi_uygula(kare, 'her_ikisi')
        return cv2.addWeighted(kare, 0.7, sobel_kare, 0.3, 0)
    return kare

def tespitleri_yumusat(tespitler, kare_id):
    if not kare_yumusatma:
        return tespitler
    yumusatilmis = []
    for tespit in tespitler:
        x1, y1, x2, y2, puan, sinif_id = tespit
        tespit_anahtari = f"{int(sinif_id)}_{int((x1+x2)/2/50)}_{int((y1+y2)/2/50)}"
        tespit_gecmisi[tespit_anahtari].append({
            'kutu': [x1, y1, x2, y2],
            'puan': puan,
            'sinif_id': sinif_id,
            'kare': kare_id
        })
        tespit_gecmisi[tespit_anahtari] = [
            d for d in tespit_gecmisi[tespit_anahtari] 
            if kare_id - d['kare'] < maks_gecmis
        ]
        if len(tespit_gecmisi[tespit_anahtari]) >= 2:
            son_tespitler = tespit_gecmisi[tespit_anahtari]
            ortalama_kutu = np.mean([d['kutu'] for d in son_tespitler], axis=0)
            ortalama_puan = np.mean([d['puan'] for d in son_tespitler])
            yumusatilmis.append([ortalama_kutu[0], ortalama_kutu[1], ortalama_kutu[2], ortalama_kutu[3], ortalama_puan, sinif_id])
    return yumusatilmis

def arayuz_bilgisi_ciz(kare, parlaklik, kontrast, esik, on_isleme_modu, kare_saniye=0):
    kaplama = kare.copy()
    arayuz_yuksekligi = 220
    cv2.rectangle(kaplama, (10, 10), (550, arayuz_yuksekligi), (0, 0, 0), -1)
    cv2.addWeighted(kaplama, 0.7, kare, 0.3, 0, kare)
    y_konumu = 30
    cv2.putText(kare, f"Parlaklik: {parlaklik:+.1f} (o/p)", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_konumu += 20
    cv2.putText(kare, f"Kontrast: {kontrast:.1f} (c/v)", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_konumu += 20
    cv2.putText(kare, f"Esik: {esik:.1f} (t)", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_konumu += 20
    cv2.putText(kare, f"Yumusatma: {'AcIK' if kare_yumusatma else 'KAPALI'} (f)", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_konumu += 20
    mod_rengi = (0, 255, 255)
    if on_isleme_modu in [9, 10, 11, 12]:
        mod_rengi = (255, 0, 255)
    cv2.putText(kare, f"on Isleme: {on_isleme_modlari[on_isleme_modu]} (m)", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, mod_rengi, 1)
    y_konumu += 20
    device_rengi = (0, 255, 0) if device == 'cuda:0' else (255, 255, 255)
    cv2.putText(kare, f"device: {device.upper()}", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, device_rengi, 1)
    if kare_saniye > 0:
        cv2.putText(kare, f"Kare/Saniye: {kare_saniye:.1f}", (150, y_konumu), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    y_konumu += 25
    cv2.putText(kare, "SISTEM ISTATISTIKLERI:", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_konumu += 20
    cpu_rengi = (0, 255, 0) if sistem_istatistikleri['cpu_yuzdesi'] < 70 else (0, 255, 255) if sistem_istatistikleri['cpu_yuzdesi'] < 90 else (0, 0, 255)
    cv2.putText(kare, f"CPU: {sistem_istatistikleri['cpu_yuzdesi']:.1f}%", (15, y_konumu), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cpu_rengi, 1)
    ram_rengi = (0, 255, 0) if sistem_istatistikleri['ram_yuzdesi'] < 70 else (0, 255, 255) if sistem_istatistikleri['ram_yuzdesi'] < 90 else (0, 0, 255)
    cv2.putText(kare, f"RAM: {sistem_istatistikleri['ram_yuzdesi']:.1f}% ({sistem_istatistikleri['ram_kullanilan_gb']:.1f}/{sistem_istatistikleri['ram_toplam_gb']:.1f}GB)", 
               (100, y_konumu), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ram_rengi, 1)
    if device == 'cuda:0':
        y_konumu += 20
        gpu_bellek_rengi = (0, 255, 0) if sistem_istatistikleri['gpu_bellek_yuzdesi'] < 70 else (0, 255, 255) if sistem_istatistikleri['gpu_bellek_yuzdesi'] < 90 else (0, 0, 255)
        cv2.putText(kare, f"GPU Bellek: {sistem_istatistikleri['gpu_bellek_yuzdesi']:.1f}%", (15, y_konumu), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_bellek_rengi, 1)
        gpu_kullanim_rengi = (0, 255, 0) if sistem_istatistikleri['gpu_kullanimi'] < 70 else (0, 255, 255) if sistem_istatistikleri['gpu_kullanimi'] < 90 else (0, 0, 255)
        cv2.putText(kare, f"GPU Kullanimi: {sistem_istatistikleri['gpu_kullanimi']:.1f}%", (150, y_konumu), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_kullanim_rengi, 1)
        if sistem_istatistikleri['gpu_sicakligi'] > 0:
            sicaklik_rengi = (0, 255, 0) if sistem_istatistikleri['gpu_sicakligi'] < 70 else (0, 255, 255) if sistem_istatistikleri['gpu_sicakligi'] < 85 else (0, 0, 255)
            cv2.putText(kare, f"GPU Sicakligi: {sistem_istatistikleri['gpu_sicakligi']:.0f}°C", (280, y_konumu), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, sicaklik_rengi, 1)

izleme_is_parcacigi = threading.Thread(target=sistem_istatistiklerini_guncelle, daemon=True)
izleme_is_parcacigi.start()

try:
    kare_sayisi = 0
    kare_saniye_sayaci = 0
    kare_saniye_zamanlayici = cv2.getTickCount()
    if device == 'cuda:0':
        print("GPU hazirlaniyor...")
        sahte_kare = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(sahte_kare, device=device, verbose=False)
        torch.cuda.synchronize()
        print("GPU hazirligi tamamlandi!")
    while True:
        ret, kare = kamera.read()
        kare_sayisi += 1
        if not ret:
            print("Hata: Kare yakalanamadi")
            break
        islenmis_kare = kareyi_on_isle(kare, parlaklik_ayari * 25.5, kontrast_ayari, mevcut_on_isleme)
        with torch.no_grad():
            sonuclar = model(islenmis_kare,
                           conf=esik,
                           iou=iou_esigi,
                           imgsz=640,
                           max_det=10,
                           device=device,
                           verbose=False)[0]
        tespitler = sonuclar.boxes.data.tolist() if len(sonuclar.boxes.data) > 0 else []
        yumusatilmis_tespitler = tespitleri_yumusat(tespitler, kare_sayisi)
        tespit_sayisi = 0
        for tespit in yumusatilmis_tespitler:
            x1, y1, x2, y2, puan, sinif_id = tespit
            if puan > esik:
                tespit_sayisi += 1
                cv2.rectangle(islenmis_kare, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                etiket = f"{sonuclar.names[int(sinif_id)].upper()}: {puan:.2f}"
                etiket_boyutu = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(islenmis_kare, 
                             (int(x1), int(y1 - etiket_boyutu[1] - 10)), 
                             (int(x1 + etiket_boyutu[0]), int(y1)), 
                             (0, 255, 0), -1)
                cv2.putText(islenmis_kare, etiket, (int(x1), int(y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        kare_saniye_sayaci += 1
        if kare_saniye_sayaci >= 30:
            kare_saniye_bitis = cv2.getTickCount()
            kare_saniye = 30.0 / ((kare_saniye_bitis - kare_saniye_zamanlayici) / cv2.getTickFrequency())
            kare_saniye_zamanlayici = kare_saniye_bitis
            kare_saniye_sayaci = 0
        else:
            kare_saniye = 0
        arayuz_bilgisi_ciz(islenmis_kare, parlaklik_ayari, kontrast_ayari, esik, 
                    mevcut_on_isleme, kare_saniye)
        if tespit_sayisi > 0:
            cv2.putText(islenmis_kare, f"Tespit edilen drone: {tespit_sayisi}", 
                       (islenmis_kare.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('GPU Hizlandirmali Drone Tespiti ve Sistem Izleme', islenmis_kare)
        tus = cv2.waitKey(1) & 0xFF
        if tus == ord('q') or tus == 27:
            break
        elif tus == ord('o'):
            parlaklik_ayari = min(parlaklik_ayari + 0.1, 2.0)
            print(f"Parlaklik: {parlaklik_ayari:+.1f}")
        elif tus == ord('p'):
            parlaklik_ayari = max(parlaklik_ayari - 0.1, -2.0)
            print(f"Parlaklik: {parlaklik_ayari:+.1f}")
        elif tus == ord('c'):
            kontrast_ayari = min(kontrast_ayari + 0.1, 3.0)
            print(f"Kontrast: {kontrast_ayari:.1f}")
        elif tus == ord('v'):
            kontrast_ayari = max(kontrast_ayari - 0.1, 0.1)
            print(f"Kontrast: {kontrast_ayari:.1f}")
        elif tus == ord('r'):
            parlaklik_ayari = 0.0
            kontrast_ayari = 1.0
            print("Parlaklik ve kontrast varsayilan degerlere sifirlandi")
        elif tus == ord('f'):
            kare_yumusatma = not kare_yumusatma
            tespit_gecmisi.clear()
            print(f"Kare yumusatma: {'AcIK' if kare_yumusatma else 'KAPALI'}")
        elif tus == ord('t'):
            esik_indeksi = (esik_indeksi + 1) % len(esik_degerleri)
            esik = esik_degerleri[esik_indeksi]
            print(f"Guven esigi: {esik:.1f}")
        elif tus == ord('m'):
            mevcut_on_isleme = (mevcut_on_isleme + 1) % len(on_isleme_modlari)
            print(f"on Isleme: {on_isleme_modlari[mevcut_on_isleme]}")
        elif tus == ord('h'):
            print("\n" + "="*60)
            print("oN IsLEME MODLARI:")
            for i, mod in on_isleme_modlari.items():
                isaretci = " -> " if i == mevcut_on_isleme else "    "
                sobel_gostergesi = " [SOBEL]" if i in [9, 10, 11, 12] else ""
                print(f"{isaretci}{i}: {mod}{sobel_gostergesi}")
            print("="*60)
            print("Sobel Kenar Tespiti Modlari:")
            print("  9: Tam Sobel kenar tespiti")
            print("  10: Sobel X-yonu (dikey kenarlar)")
            print("  11: Sobel Y-yonu (yatay kenarlar)")
            print("  12: Sobel birlesik ve orijinal (karisim)")
            print("="*60 + "\n")
        elif tus == ord('s'):
            print(f"\n{'='*60}")
            print(f"AYRINTILI SISTEM ISTATISTIKLERI")
            print(f"{'='*60}")
            print(f"CPU Kullanimi: {sistem_istatistikleri['cpu_yuzdesi']:.2f}%")
            print(f"RAM Kullanimi: {sistem_istatistikleri['ram_yuzdesi']:.2f}% ({sistem_istatistikleri['ram_kullanilan_gb']:.2f}/{sistem_istatistikleri['ram_toplam_gb']:.2f} GB)")
            if device == 'cuda:0':
                print(f"GPU Bellek: {sistem_istatistikleri['gpu_bellek_yuzdesi']:.2f}%")
                print(f"GPU Kullanimi: {sistem_istatistikleri['gpu_kullanimi']:.2f}%")
                if sistem_istatistikleri['gpu_sicakligi'] > 0:
                    print(f"GPU Sicakligi: {sistem_istatistikleri['gpu_sicakligi']:.1f}°C")
                print(f"CUDA Surumu: {torch.version.cuda}")
                print(f"GPU Cihazi: {torch.cuda.get_device_name(0)}")
            print(f"{'='*60}\n")

except KeyboardInterrupt:
    print("\nKullanici tarafindan kesildi")

finally:
    if device == 'cuda:0':
        torch.cuda.empty_cache()
        print("GPU bellek onbellegi temizlendi")
    kamera.release()
    cv2.destroyAllWindows()
    print("Web kamerasi serbest birakildi ve pencereler kapatildi")