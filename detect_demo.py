"""
Pothole Detection - Full Inference Demo
=========================================
Shows the complete pipeline: photo -> preprocess -> TFLite model -> postprocess -> draw boxes

This is a REFERENCE implementation. Your Android teammate should replicate this
same logic in Kotlin using the TensorFlow Lite Android library. The steps
(preprocess, confidence filter, NMS, draw) are identical regardless of language.

Usage:
    python detect_demo.py --model best_float16.tflite --image sample_pothole.jpg --conf 0.5

Requires: pip install tensorflow opencv-python-headless numpy
"""
import argparse
import numpy as np
import cv2
import tensorflow as tf


def preprocess(image_path, input_size=640):
    """
    Step 1-2: Load image, resize to model's expected input, normalize pixel values.
    Returns the preprocessed tensor + original image + scale factors (needed to
    map detected boxes back to original image size).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]

    # Resize to model input size (640x640) - this is what YOLOv8 expects
    resized = cv2.resize(img, (input_size, input_size))

    # BGR -> RGB (OpenCV loads as BGR, model expects RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize 0-255 -> 0-1 and convert to float32
    normalized = rgb.astype(np.float32) / 255.0

    # Add batch dimension: (640, 640, 3) -> (1, 640, 640, 3)
    input_tensor = np.expand_dims(normalized, axis=0)

    return input_tensor, img, orig_w, orig_h


def run_inference(model_path, input_tensor):
    """
    Step 3-4: Load TFLite model, run it on the preprocessed image, get raw output.
    Raw output shape: (1, 5, 8400) -> [x_center, y_center, width, height, confidence]
    for 8400 candidate boxes (most are noise/overlaps - postprocess() cleans this up).
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output


def postprocess(raw_output, orig_w, orig_h, input_size=640, conf_threshold=0.5, iou_threshold=0.45):
    """
    Step 5: Filter out low-confidence boxes, apply Non-Max Suppression (NMS) to
    remove duplicate overlapping boxes, and rescale coordinates back to the
    original image size.
    """
    # raw_output shape: (1, 5, 8400) -> transpose to (8400, 5)
    predictions = raw_output[0].T  # now (8400, 5): [x, y, w, h, conf]

    boxes = []
    confidences = []

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    for pred in predictions:
        x_center, y_center, width, height, conf = pred

        if conf < conf_threshold:
            continue

        # Convert center-format to corner-format (x1, y1, x2, y2), scaled to input_size
        x1 = (x_center - width / 2) * input_size
        y1 = (y_center - height / 2) * input_size
        x2 = (x_center + width / 2) * input_size
        y2 = (y_center + height / 2) * input_size

        # Rescale to original image dimensions
        x1, x2 = x1 * scale_x, x2 * scale_x
        y1, y2 = y1 * scale_y, y2 * scale_y

        boxes.append([x1, y1, x2 - x1, y2 - y1])  # cv2.dnn.NMSBoxes wants [x, y, w, h]
        confidences.append(float(conf))

    if not boxes:
        return []

    # NMS: removes duplicate/overlapping boxes for the same pothole
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    final_detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            final_detections.append({
                "box": (int(x), int(y), int(w), int(h)),
                "confidence": confidences[i]
            })

    return final_detections


def estimate_severity(box, image_shape):
    """
    Severity score (0-10) based on:
      1. Area ratio   - bigger box relative to image = more severe
      2. Shape factor - round/wide boxes (deep potholes) score higher than
                         thin elongated boxes (surface cracks), since cracks
                         are generally less immediately dangerous than potholes
      3. Position      - boxes nearer the vertical center of the frame (i.e.
                         closer to where a vehicle's wheels/path typically is)
                         are weighted slightly higher than boxes near the edges

    NOTE: thresholds/weights below are reasonable starting points, not
    calibrated against ground truth. For the hackathon demo this is fine -
    if you get time, sanity-check scores against a few real images and adjust
    SEVERITY_AREA_WEIGHT / SEVERITY_SHAPE_WEIGHT / SEVERITY_POSITION_WEIGHT below.
    """
    img_h, img_w = image_shape[:2]
    image_area = img_w * img_h
    x, y, w, h = box

    # --- 1. Area score (0-10 scale, saturates around 8% of image area) ---
    area_ratio = (w * h) / image_area
    area_score = min(10, (area_ratio / 0.08) * 10)

    # --- 2. Shape score: round/blob-like (aspect ratio near 1) scores higher
    #         than thin elongated shapes (aspect ratio far from 1, i.e. cracks) ---
    aspect_ratio = w / h if h > 0 else 1
    # distance of aspect ratio from 1 (perfectly square/round) -> 0 is best (roundest)
    elongation = abs(1 - min(aspect_ratio, 1 / aspect_ratio if aspect_ratio > 0 else 1))
    shape_score = max(0, 10 * (1 - elongation))  # roundest boxes -> 10, thinnest -> 0

    # --- 3. Position score: boxes near vertical center score slightly higher
    #         (assumes camera roughly centered on the road/wheel path) ---
    box_center_y = y + h / 2
    vertical_center = img_h / 2
    center_distance_ratio = abs(box_center_y - vertical_center) / (img_h / 2)
    position_score = max(0, 10 * (1 - center_distance_ratio))

    # --- Weighted combination ---
    SEVERITY_AREA_WEIGHT = 0.6
    SEVERITY_SHAPE_WEIGHT = 0.25
    SEVERITY_POSITION_WEIGHT = 0.15

    final_score = (
        area_score * SEVERITY_AREA_WEIGHT +
        shape_score * SEVERITY_SHAPE_WEIGHT +
        position_score * SEVERITY_POSITION_WEIGHT
    )

    return round(min(10, max(0, final_score)), 1)


def draw_detections(image, detections):
    """
    Step 6: Draw the final bounding boxes + confidence + severity on the image.
    """
    for det in detections:
        x, y, w, h = det["box"]
        conf = det["confidence"]
        severity = estimate_severity(det["box"], image.shape)

        color = (0, 0, 255)  # red box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = f"Pothole {conf:.2f} | Severity: {severity}/10"
        cv2.putText(image, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--out", default="detected_output.jpg", help="Output image path")
    args = parser.parse_args()

    print(f"Loading and preprocessing: {args.image}")
    input_tensor, orig_img, orig_w, orig_h = preprocess(args.image)

    print(f"Running inference with: {args.model}")
    raw_output = run_inference(args.model, input_tensor)

    print("Postprocessing (confidence filter + NMS)...")
    detections = postprocess(raw_output, orig_w, orig_h, conf_threshold=args.conf)

    print(f"Found {len(detections)} pothole(s)")
    for i, det in enumerate(detections):
        print(f"  #{i+1}: box={det['box']}, confidence={det['confidence']:.2f}")

    result_img = draw_detections(orig_img, detections)
    cv2.imwrite(args.out, result_img)
    print(f"Saved annotated image to: {args.out}")


if __name__ == "__main__":
    main()
