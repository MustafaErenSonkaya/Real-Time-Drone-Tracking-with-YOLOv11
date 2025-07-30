import cv2
import os
from pathlib import Path

def extract_screenshots(video_path, output_dir="screenshots", interval=0.3):
    """
    Extract screenshots from a video at specified intervals.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save screenshots
        interval (float): Time interval between screenshots in seconds
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Taking screenshots every {interval} seconds")
    
    # Calculate frame interval
    frame_interval = int(fps * interval)
    
    frame_count = 0
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save screenshot at specified intervals
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            filename = f"screenshot_{screenshot_count:04d}_t{timestamp:.1f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, frame)
            print(f"Saved: {filename}")
            screenshot_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"\nExtraction complete! Saved {screenshot_count} screenshots to '{output_dir}' directory")

def main():
    # Example usage
    video_path = input("Enter the path to your video file: ").strip()
    
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return
    
    # Optional: customize output directory and interval
    output_dir = input("Enter output directory (or press Enter for 'screenshots'): ").strip()
    if not output_dir:
        output_dir = "screenshots"
    
    try:
        interval = float(input("Enter interval in seconds (or press Enter for 0.3): ").strip() or "0.3")
    except ValueError:
        interval = 0.3
    
    extract_screenshots(video_path, output_dir, interval)

if __name__ == "__main__":
    main()