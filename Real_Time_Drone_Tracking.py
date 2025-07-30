# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:49:34 2025

@author: musta
"""

# -*- coding: utf-8 -*-
import os
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import torch
import psutil  # For CPU and RAM monitoring
import time
import threading

# Check CUDA availability and set device
if torch.cuda.is_available():
    device = 'cuda:0'  # Use first CUDA device (RTX 3050 Ti)
    print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("CUDA not available, using CPU")

# Load the custom YOLO model and move to GPU
model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'last.pt')
model = YOLO(model_path)  # load a custom model
model.to(device)  # Move model to GPU

# Detection parameters
threshold = 0.3
iou_threshold = 0.5
brightness_adjust = 0.0  # Manual brightness adjustment
contrast_adjust = 1.0    # Manual contrast adjustment

# Image preprocessing modes - Added Sobel filtering options
preprocessing_modes = {
    0: "None",
    1: "Histogram Equalization", 
    2: "CLAHE (Adaptive)",
    3: "Gaussian Blur + Sharpen",
    4: "Edge Enhancement",
    5: "Noise Reduction",
    6: "Gamma Correction",
    7: "Color Space (HSV)",
    8: "Contrast Stretching",
    9: "Sobel Edge Detection",
    10: "Sobel X-Direction",
    11: "Sobel Y-Direction",
    12: "Sobel Combined + Original"
}
current_preprocessing = 0

# Tracking for smoother detections
detection_history = defaultdict(list)
max_history = 5  # Keep last 5 frames for smoothing

# System monitoring variables
system_stats = {
    'cpu_percent': 0.0,
    'ram_percent': 0.0,
    'ram_used_gb': 0.0,
    'ram_total_gb': 0.0,
    'gpu_memory_percent': 0.0,
    'gpu_utilization': 0.0,
    'gpu_temp': 0.0
}

def update_system_stats():
    """Background thread to update system statistics"""
    global system_stats
    while True:
        try:
            # CPU usage
            system_stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            
            # RAM usage
            ram = psutil.virtual_memory()
            system_stats['ram_percent'] = ram.percent
            system_stats['ram_used_gb'] = ram.used / (1024**3)
            system_stats['ram_total_gb'] = ram.total / (1024**3)
            
            # GPU stats (if CUDA available)
            if device == 'cuda:0':
                try:
                    # Memory usage
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_total = torch.cuda.get_device_properties(0).total_memory
                    system_stats['gpu_memory_percent'] = (memory_allocated / memory_total) * 100
                    
                    # Estimate GPU utilization (rough approximation)
                    system_stats['gpu_utilization'] = min(100, system_stats['gpu_memory_percent'] * 1.5)
                    
                    # Try to get GPU temperature (may not work on all systems)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                        system_stats['gpu_temp'] = temp
                    except:
                        system_stats['gpu_temp'] = 0.0  # Temperature not available
                        
                except Exception as e:
                    print(f"Error getting GPU stats: {e}")
            
            time.sleep(0.5)  # Update every 500ms
            
        except Exception as e:
            print(f"Error in system monitoring: {e}")
            time.sleep(1)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Set camera resolution (optional - adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting GPU-accelerated real-time drone detection with system monitoring...")
print(f"Device: {device.upper()}")
if device == 'cuda:0':
    print("GPU acceleration enabled for YOLO inference!")
print("Controls:")
print("'q' or ESC - Quit")
print("'o' / 'p' - Increase/Decrease brightness (±0.1)")
print("'c' / 'v' - Increase/Decrease contrast (±0.1)")
print("'r' - Reset brightness and contrast")
print("'f' - Toggle frame smoothing")
print("'t' - Cycle through confidence thresholds")
print("'m' - Cycle through preprocessing modes (includes Sobel filters)")
print("'h' - Show preprocessing help")
print("'s' - Show detailed system stats")

# Control variables
frame_smoothing = True
threshold_values = [0.2, 0.3, 0.4, 0.5, 0.6]
threshold_index = 1  # Start with 0.3

def apply_sobel_filter(frame, direction='both'):
    """Apply Sobel edge detection filter"""
    # Convert to grayscale first
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if direction == 'x':
        # Sobel X (vertical edges)
        sobel = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    elif direction == 'y':
        # Sobel Y (horizontal edges)
        sobel = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    else:
        # Combined Sobel (both directions)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Convert to uint8 and normalize
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    
    # Convert back to BGR for consistency
    return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

def preprocess_frame(frame, brightness=0.0, contrast=1.0, mode=0):
    """Apply various preprocessing techniques to improve detection"""
    
    # First apply manual brightness and contrast
    if brightness != 0.0 or contrast != 1.0:
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    
    # Apply selected preprocessing mode
    if mode == 0:  # None
        return frame
    
    elif mode == 1:  # Histogram Equalization
        # Convert to YUV and equalize Y channel
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    elif mode == 2:  # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Convert to LAB color space and apply CLAHE to L channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif mode == 3:  # Gaussian Blur + Sharpen
        # First blur to reduce noise, then sharpen
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
        return cv2.filter2D(blurred, -1, kernel)
    
    elif mode == 4:  # Edge Enhancement
        # Enhance edges using unsharp masking
        gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
        return cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    
    elif mode == 5:  # Noise Reduction
        # Bilateral filter to reduce noise while preserving edges
        return cv2.bilateralFilter(frame, 9, 75, 75)
    
    elif mode == 6:  # Gamma Correction
        # Apply gamma correction for better exposure
        gamma = 1.2  # Adjust gamma value
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)
    
    elif mode == 7:  # Color Space Enhancement (HSV)
        # Enhance saturation and value in HSV space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1] *= 1.2  # Increase saturation
        hsv[:,:,2] *= 1.1  # Increase value/brightness
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    elif mode == 8:  # Contrast Stretching
        # Stretch contrast using percentiles
        p2, p98 = np.percentile(frame, (2, 98))
        return cv2.convertScaleAbs(frame, alpha=255.0/(p98-p2), beta=-p2*255.0/(p98-p2))
    
    elif mode == 9:  # Sobel Edge Detection (Both directions)
        return apply_sobel_filter(frame, 'both')
    
    elif mode == 10:  # Sobel X-Direction (Vertical edges)
        return apply_sobel_filter(frame, 'x')
    
    elif mode == 11:  # Sobel Y-Direction (Horizontal edges)
        return apply_sobel_filter(frame, 'y')
    
    elif mode == 12:  # Sobel Combined + Original
        # Apply Sobel and blend with original frame
        sobel_frame = apply_sobel_filter(frame, 'both')
        # Blend: 70% original + 30% Sobel edges
        return cv2.addWeighted(frame, 0.7, sobel_frame, 0.3, 0)
    
    return frame

def smooth_detections(detections, frame_id):
    """Smooth detections across multiple frames to reduce flickering"""
    if not frame_smoothing:
        return detections
    
    smoothed = []
    for detection in detections:
        x1, y1, x2, y2, score, class_id = detection
        detection_key = f"{int(class_id)}_{int((x1+x2)/2/50)}_{int((y1+y2)/2/50)}"
        
        # Add to history
        detection_history[detection_key].append({
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'class_id': class_id,
            'frame': frame_id
        })
        
        # Keep only recent detections
        detection_history[detection_key] = [
            d for d in detection_history[detection_key] 
            if frame_id - d['frame'] < max_history
        ]
        
        # If we have enough consistent detections, include it
        if len(detection_history[detection_key]) >= 2:
            # Average the bounding box for stability
            recent_detections = detection_history[detection_key]
            avg_bbox = np.mean([d['bbox'] for d in recent_detections], axis=0)
            avg_score = np.mean([d['score'] for d in recent_detections])
            
            smoothed.append([avg_bbox[0], avg_bbox[1], avg_bbox[2], avg_bbox[3], avg_score, class_id])
    
    return smoothed

def draw_ui_info(frame, brightness, contrast, threshold, preprocessing_mode, fps=0):
    """Draw UI information and system stats on frame"""
    # Make UI panel larger for system stats
    overlay = frame.copy()
    ui_height = 220
    cv2.rectangle(overlay, (10, 10), (550, ui_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # UI text
    y_offset = 30
    cv2.putText(frame, f"Brightness: {brightness:+.1f} (o/p)", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, f"Contrast: {contrast:.1f} (c/v)", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, f"Threshold: {threshold:.1f} (t)", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, f"Smoothing: {'ON' if frame_smoothing else 'OFF'} (f)", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    
    # Color code Sobel modes
    mode_color = (0, 255, 255)  # Default yellow
    if preprocessing_mode in [9, 10, 11, 12]:  # Sobel modes
        mode_color = (255, 0, 255)  # Magenta for Sobel modes
    
    cv2.putText(frame, f"Preprocessing: {preprocessing_modes[preprocessing_mode]} (m)", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
    y_offset += 20
    
    # Device and FPS info
    device_color = (0, 255, 0) if device == 'cuda:0' else (255, 255, 255)
    cv2.putText(frame, f"Device: {device.upper()}", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, device_color, 1)
    
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (150, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # System Stats Section
    y_offset += 25
    cv2.putText(frame, "SYSTEM STATS:", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    y_offset += 20
    
    # CPU Usage
    cpu_color = (0, 255, 0) if system_stats['cpu_percent'] < 70 else (0, 255, 255) if system_stats['cpu_percent'] < 90 else (0, 0, 255)
    cv2.putText(frame, f"CPU: {system_stats['cpu_percent']:.1f}%", (15, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, cpu_color, 1)
    
    # RAM Usage
    ram_color = (0, 255, 0) if system_stats['ram_percent'] < 70 else (0, 255, 255) if system_stats['ram_percent'] < 90 else (0, 0, 255)
    cv2.putText(frame, f"RAM: {system_stats['ram_percent']:.1f}% ({system_stats['ram_used_gb']:.1f}/{system_stats['ram_total_gb']:.1f}GB)", 
               (100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ram_color, 1)
    
    # GPU Stats (if available)
    if device == 'cuda:0':
        y_offset += 20
        gpu_mem_color = (0, 255, 0) if system_stats['gpu_memory_percent'] < 70 else (0, 255, 255) if system_stats['gpu_memory_percent'] < 90 else (0, 0, 255)
        cv2.putText(frame, f"GPU Mem: {system_stats['gpu_memory_percent']:.1f}%", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_mem_color, 1)
        
        gpu_util_color = (0, 255, 0) if system_stats['gpu_utilization'] < 70 else (0, 255, 255) if system_stats['gpu_utilization'] < 90 else (0, 0, 255)
        cv2.putText(frame, f"GPU Util: {system_stats['gpu_utilization']:.1f}%", (150, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gpu_util_color, 1)
        
        # GPU Temperature (if available)
        if system_stats['gpu_temp'] > 0:
            temp_color = (0, 255, 0) if system_stats['gpu_temp'] < 70 else (0, 255, 255) if system_stats['gpu_temp'] < 85 else (0, 0, 255)
            cv2.putText(frame, f"GPU Temp: {system_stats['gpu_temp']:.0f}°C", (280, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 1)

# Start system monitoring thread
monitoring_thread = threading.Thread(target=update_system_stats, daemon=True)
monitoring_thread.start()

try:
    frame_count = 0
    fps_counter = 0
    fps_timer = cv2.getTickCount()
    
    # Warmup GPU if using CUDA
    if device == 'cuda:0':
        print("Warming up GPU...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_frame, device=device, verbose=False)
        torch.cuda.synchronize()  # Wait for GPU operations to complete
        print("GPU warmup complete!")
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Apply manual adjustments and preprocessing
        processed_frame = preprocess_frame(frame, brightness_adjust * 25.5, contrast_adjust, current_preprocessing)
        
        # Run YOLOv8 inference with GPU acceleration
        with torch.no_grad():  # Disable gradient computation for inference
            results = model(processed_frame,
                           conf=threshold,           # Confidence threshold
                           iou=iou_threshold,        # IoU threshold for NMS
                           imgsz=640,                # Image size for inference
                           max_det=10,               # Maximum detections per image
                           device=device,            # Explicitly specify device
                           verbose=False)[0]
        
        # Get detections and apply smoothing
        detections = results.boxes.data.tolist() if len(results.boxes.data) > 0 else []
        smoothed_detections = smooth_detections(detections, frame_count)
        
        # Draw bounding boxes and labels
        detection_count = 0
        for detection in smoothed_detections:
            x1, y1, x2, y2, score, class_id = detection
            
            if score > threshold:
                detection_count += 1
                
                # Draw bounding box with thicker line for better visibility
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                
                # Draw label with background for better readability
                label = f"{results.names[int(class_id)].upper()}: {score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Label background
                cv2.rectangle(processed_frame, 
                             (int(x1), int(y1 - label_size[1] - 10)), 
                             (int(x1 + label_size[0]), int(y1)), 
                             (0, 255, 0), -1)
                
                # Label text
                cv2.putText(processed_frame, label, (int(x1), int(y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 30:  # Update FPS every 30 frames
            fps_timer_end = cv2.getTickCount()
            fps = 30.0 / ((fps_timer_end - fps_timer) / cv2.getTickFrequency())
            fps_timer = fps_timer_end
            fps_counter = 0
        else:
            fps = 0
        
        # Draw UI information with system stats
        draw_ui_info(processed_frame, brightness_adjust, contrast_adjust, threshold, 
                    current_preprocessing, fps)
        
        # Show detection count
        if detection_count > 0:
            cv2.putText(processed_frame, f"Drones detected: {detection_count}", 
                       (processed_frame.shape[1] - 250, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the frame with detections
        cv2.imshow('GPU-Accelerated Drone Detection with System Monitoring', processed_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' key or ESC key
            break
        elif key == ord('o'):  # Increase brightness
            brightness_adjust = min(brightness_adjust + 0.1, 2.0)
            print(f"Brightness: {brightness_adjust:+.1f}")
        elif key == ord('p'):  # Decrease brightness
            brightness_adjust = max(brightness_adjust - 0.1, -2.0)
            print(f"Brightness: {brightness_adjust:+.1f}")
        elif key == ord('c'):  # Increase contrast
            contrast_adjust = min(contrast_adjust + 0.1, 3.0)
            print(f"Contrast: {contrast_adjust:.1f}")
        elif key == ord('v'):  # Decrease contrast
            contrast_adjust = max(contrast_adjust - 0.1, 0.1)
            print(f"Contrast: {contrast_adjust:.1f}")
        elif key == ord('r'):  # Reset brightness and contrast
            brightness_adjust = 0.0
            contrast_adjust = 1.0
            print("Brightness and contrast reset to defaults")
        elif key == ord('f'):  # Toggle frame smoothing
            frame_smoothing = not frame_smoothing
            detection_history.clear()  # Clear history when toggling
            print(f"Frame smoothing: {'ON' if frame_smoothing else 'OFF'}")
        elif key == ord('t'):  # Cycle through thresholds
            threshold_index = (threshold_index + 1) % len(threshold_values)
            threshold = threshold_values[threshold_index]
            print(f"Confidence threshold: {threshold:.1f}")
        elif key == ord('m'):  # Cycle through preprocessing modes
            current_preprocessing = (current_preprocessing + 1) % len(preprocessing_modes)
            print(f"Preprocessing: {preprocessing_modes[current_preprocessing]}")
        elif key == ord('h'):  # Show preprocessing help
            print("\n" + "="*60)
            print("PREPROCESSING MODES:")
            for i, mode in preprocessing_modes.items():
                marker = " -> " if i == current_preprocessing else "    "
                sobel_indicator = " [SOBEL]" if i in [9, 10, 11, 12] else ""
                print(f"{marker}{i}: {mode}{sobel_indicator}")
            print("="*60)
            print("Sobel Edge Detection Modes:")
            print("  9: Full Sobel edge detection")
            print("  10: Sobel X-direction (vertical edges)")
            print("  11: Sobel Y-direction (horizontal edges)")
            print("  12: Sobel combined with original (blend)")
            print("="*60 + "\n")
        elif key == ord('s'):  # Show detailed system stats
            print(f"\n{'='*60}")
            print(f"DETAILED SYSTEM STATISTICS")
            print(f"{'='*60}")
            print(f"CPU Usage: {system_stats['cpu_percent']:.2f}%")
            print(f"RAM Usage: {system_stats['ram_percent']:.2f}% ({system_stats['ram_used_gb']:.2f}/{system_stats['ram_total_gb']:.2f} GB)")
            if device == 'cuda:0':
                print(f"GPU Memory: {system_stats['gpu_memory_percent']:.2f}%")
                print(f"GPU Utilization: {system_stats['gpu_utilization']:.2f}%")
                if system_stats['gpu_temp'] > 0:
                    print(f"GPU Temperature: {system_stats['gpu_temp']:.1f}°C")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"{'='*60}\n")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Clean up GPU memory
    if device == 'cuda:0':
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed")