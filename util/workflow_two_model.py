import gxipy as gx
import numpy as np
import cv2
from ultralytics import YOLO
import time

# === Camera Calibration Parameters ===
# You need to calibrate these values based on your setup
# Method 1: Use a reference object of known size
# Method 2: Use camera focal length and distance to object
PIXELS_PER_MM = 11.4  # Example: 2.5 pixels = 1mm (ADJUST THIS FOR YOUR SETUP)

# Alternative method using camera parameters (uncomment if you know these values)
# FOCAL_LENGTH_MM = 16  # Your lens focal length in mm
# SENSOR_WIDTH_MM = 5.76  # Your camera sensor width in mm
# IMAGE_WIDTH_PIXELS = 1920  # Your image width in pixels
# DISTANCE_TO_OBJECT_MM = 100  # Distance from camera to object in mm

def pixels_to_mm(pixels):
    """Convert pixels to millimeters using calibration factor"""
    return pixels / PIXELS_PER_MM

def get_position_and_size_mm(x1, y1, x2, y2):
    """Calculate position and size in millimeters"""
    center_x_px = (x1 + x2) // 2
    center_y_px = (y1 + y2) // 2
    width_px = x2 - x1
    height_px = y2 - y1
    
    # Convert to millimeters
    center_x_mm = pixels_to_mm(center_x_px)
    center_y_mm = pixels_to_mm(center_y_px)
    width_mm = pixels_to_mm(width_px)
    height_mm = pixels_to_mm(height_px)
    
    return center_x_mm, center_y_mm, width_mm, height_mm

# === Load Multiple YOLO models ===
models = {
    "capacitor": {
        "model": YOLO("capacitor_yollo11_robo.pt"),  # Your existing capacitor model
        "color": (0, 255, 0),  # Green for capacitors
        "confidence": 0.5
    },
    "IC": {
        "model": YOLO("IC_colab.pt"),  # Replace with your workflow model path
        "color": (255, 0, 0),  # Blue for workflow objects
        "confidence": 0.5
    }
    # Add more models here as needed
    # "another_object": {
    #     "model": YOLO("another_model.pt"),
    #     "color": (0, 0, 255),  # Red
    #     "confidence": 0.4
    # }
}

print("🔄 Loading models...")
for model_name, model_info in models.items():
    print(f"✅ Loaded {model_name} model")

# === Initialize Daheng camera ===
device_manager = gx.DeviceManager()
dev_num = device_manager.update_device_list()
if dev_num == 0:
    raise Exception("❌ No Daheng camera found!")

cam = device_manager.open_device_by_index(1)
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
cam.ExposureTime.set(30000.0)
cam.stream_on()

cv2.namedWindow("🟡 Multi-Model YOLO Detection", cv2.WINDOW_NORMAL)

print("🟢 Multi-model YOLO detection started. Press 'q' to quit.")
print(f"📏 Using calibration: {PIXELS_PER_MM} pixels per mm")
print(f"🤖 Active models: {', '.join(models.keys())}")

def draw_detection_info(annotated_frame, x1, y1, x2, y2, class_name, confidence, 
                       center_x_mm, center_y_mm, width_mm, height_mm, color, model_name):
    """Draw detection information on the frame"""
    
    # Draw bounding box
    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

    # Prepare text labels with proper spacing
    confidence_text = f"[{model_name}] {class_name}: {confidence:.2f}"
    position_text = f"Pos: ({center_x_mm:.1f}, {center_y_mm:.1f})mm"
    size_text = f"Size: {width_mm:.1f}×{height_mm:.1f}mm"

    # Calculate text positions with proper spacing
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    
    # Get text sizes for proper positioning
    (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
    (pos_w, pos_h), _ = cv2.getTextSize(position_text, font, font_scale, thickness)
    (size_w, size_h), _ = cv2.getTextSize(size_text, font, font_scale, thickness)
    
    # Calculate the maximum text width and total height needed
    max_text_width = max(conf_w, pos_w, size_w)
    total_text_height = conf_h + pos_h + size_h + 20  # 20 for spacing
    
    # Determine text position - try to place it to the right of the object
    img_height, img_width = annotated_frame.shape[:2]
    
    text_offset = 10  # Distance from bounding box
    text_x = x2 + text_offset  # Try right side first
    text_y = y1
    
    # Check if text fits on the right side
    if text_x + max_text_width + 10 > img_width:
        # Try left side
        text_x = x1 - max_text_width - text_offset
        if text_x < 0:
            # Place below the object
            text_x = x1
            text_y = y2 + text_offset + conf_h
            
            # If below goes off screen, place above
            if text_y + total_text_height > img_height:
                text_y = y1 - total_text_height - text_offset

    # Ensure text doesn't go off screen vertically
    if text_y < conf_h + 10:
        text_y = conf_h + 10
    elif text_y + total_text_height > img_height:
        text_y = img_height - total_text_height - 10

    # Ensure text doesn't go off screen horizontally
    if text_x < 0:
        text_x = 5
    elif text_x + max_text_width > img_width:
        text_x = img_width - max_text_width - 5

    # Draw text background for better readability
    bg_x1 = max(0, text_x - 5)
    bg_y1 = max(0, text_y - conf_h - 5)
    bg_x2 = min(img_width, text_x + max_text_width + 10)
    bg_y2 = min(img_height, text_y + pos_h + size_h + 15)
    
    cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    
    # Draw confidence text
    cv2.putText(annotated_frame, confidence_text,
               (text_x, text_y),
               font, font_scale, (0, 255, 255), thickness)  # Yellow
    
    # Draw position text (with spacing)
    cv2.putText(annotated_frame, position_text,
               (text_x, text_y + conf_h + 8),
               font, font_scale, (255, 255, 0), thickness)  # Cyan
    
    # Draw size text (with spacing)
    cv2.putText(annotated_frame, size_text,
               (text_x, text_y + conf_h + pos_h + 16),
               font, font_scale, (255, 0, 255), thickness)  # Magenta

    # Draw center point
    center_x_px = (x1 + x2) // 2
    center_y_px = (y1 + y2) // 2
    cv2.circle(annotated_frame, (center_x_px, center_y_px), 3, color, -1)

    # Draw a line connecting the text to the object (optional)
    text_center_x = text_x + max_text_width // 2
    text_center_y = text_y + total_text_height // 2
    cv2.line(annotated_frame, (center_x_px, center_y_px), 
            (text_center_x, text_center_y), (128, 128, 128), 1)

try:
    while True:
        raw_image = cam.data_stream[0].get_image(timeout=1000)
        if raw_image is None:
            print("⚠️ Failed to get image.")
            continue

        img = raw_image.get_numpy_array()
        if img is None:
            print("⚠️ Failed to convert image.")
            continue

        # Resize image (optional)
        scale = 0.5
        resized_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        # Convert grayscale to BGR for YOLO
        if len(resized_img.shape) == 2:
            color_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        else:
            color_img = resized_img

        # Start with the original image for custom annotations
        annotated_frame = color_img.copy()

        # Process each model
        total_detections = 0
        for model_name, model_info in models.items():
            model = model_info["model"]
            color = model_info["color"]
            confidence_threshold = model_info["confidence"]
            
            # Perform detection
            results = model.predict(source=color_img, conf=confidence_threshold, verbose=False)
            boxes = results[0].boxes
            
            # Process each detection from this model
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    class_name = model.names[class_id]
                    confidence = box.conf[0].item()

                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box corners
                    
                    # Get measurements in millimeters
                    center_x_mm, center_y_mm, width_mm, height_mm = get_position_and_size_mm(x1, y1, x2, y2)

                    # Print position info in mm
                    print(f"🟣 [{model_name}] Detected {class_name}")
                    print(f"   📍 Position: ({center_x_mm:.1f}mm, {center_y_mm:.1f}mm)")
                    print(f"   📐 Size: {width_mm:.1f}mm × {height_mm:.1f}mm")
                    print(f"   🎯 Confidence: {confidence:.2f}")
                    print("   " + "-"*40)

                    # Draw detection info
                    draw_detection_info(annotated_frame, x1, y1, x2, y2, class_name, confidence,
                                      center_x_mm, center_y_mm, width_mm, height_mm, color, model_name)
                    
                    total_detections += 1

        # Add detection summary to the frame
        summary_text = f"Total Detections: {total_detections}"
        cv2.putText(annotated_frame, summary_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add model legend
        legend_y = 60
        for i, (model_name, model_info) in enumerate(models.items()):
            legend_text = f"{model_name.upper()}"
            cv2.putText(annotated_frame, legend_text, (10, legend_y + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_info["color"], 2)

        # Display image
        cv2.imshow("🟡 Multi-Model YOLO Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Interrupted by user.")

# Cleanup
cam.stream_off()
cam.close_device()
cv2.destroyAllWindows()

print("✅ Stream closed.")

# === CALIBRATION INSTRUCTIONS ===
print("\n📋 CALIBRATION INSTRUCTIONS:")
print("1. Place an object of known size in your camera view")
print("2. Measure the object size in pixels in the image")
print("3. Calculate: PIXELS_PER_MM = pixels_measured / actual_size_mm")
print("4. Update the PIXELS_PER_MM value at the top of this script")
print("\nExample: If a 10mm object appears as 25 pixels:")
print("PIXELS_PER_MM = 25 / 10 = 2.5")

print("\n🤖 MODEL CONFIGURATION:")
print("1. Add your workflow model file path in the 'workflow' section")
print("2. Adjust confidence thresholds for each model as needed")
print("3. Customize colors for different object types")
print("4. Add more models by extending the 'models' dictionary")