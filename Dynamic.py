import streamlit as st
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO

def upload_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        return image
    else:
        return None

def process_dynamicImage(image):
    if image is None:
        st.error("Failed to load image.")
        return

    max_width = 800
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        image = cv2.resize(image, new_size)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    min_area_threshold = 100

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue

        found_group = False
        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            if abs(mean_area - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                found_group = True
                break

        if not found_group:
            size_groups.append((area, 1))

    colors = [tuple(random.sample(range(256), 3)) for _ in range(len(size_groups))]

    output_image = image.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue

        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            if abs(mean_area - area) <= 800:
                cv2.drawContours(output_image, [contour], -1, colors[idx], 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0
                label = chr(65 + idx)
                cv2.putText(output_image, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                break

    y0, dy = 30, 30
    for i, (group_area, count) in enumerate(size_groups):
        mean_area = group_area / count
        size_label = f"Size {chr(65 + i)}: {count} (Mean Area: {mean_area:.1f})"
        cv2.putText(output_image, size_label, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    st.subheader("Dynamic Object Classification")
    fig, ax = plt.subplots()
    ax.imshow(output_image_rgb)
    ax.axis('off')
    ax.set_title("Dynamic Object Classification")
    st.pyplot(fig)

def dynamicDetection():
    image = upload_image()
    process_dynamicImage(image)

stable_colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 128),   # Purple
    (128, 128, 0)    # Olive
]

def process_liveDynamic(frame):
    max_width = 800
    height, width = frame.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        frame = cv2.resize(frame, new_size)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    min_area_threshold = 100

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue

        found_group = False
        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            if abs(mean_area - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                found_group = True
                break

        if not found_group:
            size_groups.append((area, 1))

    output_frame = frame.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area_threshold:
            continue

        for idx, (group_area, count) in enumerate(size_groups):
            mean_area = group_area / count
            if abs(mean_area - area) <= 800:
                color = stable_colors[idx % len(stable_colors)]
                cv2.drawContours(output_frame, [contour], -1, color, 2)
                break

    y0, dy = 50, 30
    for i, (group_area, count) in enumerate(size_groups):
        mean_area = group_area / count
        size_label = f"Size {chr(65 + i)}: {count} (Mean Area: {mean_area:.1f})"
        color = stable_colors[i % len(stable_colors)]
        cv2.putText(output_frame, size_label, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return output_frame

def liveDynamic():
    st.subheader("Live Dynamic Shape Detection (Webcam)")
    WEBCAM_ID = st.number_input("Enter Webcam ID (usually 0 or 1):", value=1, step=1, key="live_dynamic_webcam_id")
    live_detection = st.checkbox("Start Live Dynamic Shape Detection")
    video_placeholder = st.empty()

    if live_detection:
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            st.error("Failed to open webcam. Please check the webcam ID.")
            return

        while live_detection:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to capture frame from webcam.")
                break

            processed_frame = process_liveDynamic(frame)
            video_placeholder.image(processed_frame, channels="BGR")

            if st.button("Stop Live Dynamic Detection"):
                live_detection = False

        cap.release()
        cv2.destroyAllWindows()

def main():
    st.title("Dynamic Objects Counting Using Image Processing")

    # Load and display an image
    st.image("DynamicScrew.png", caption="Detection Dynamic Objects", use_container_width=True)  # Updated parameter

    operation_type = st.sidebar.selectbox(
        "Choose an Operation:",
        (
            "Select an option",
            "Upload Image for Dynamic Shape Classification",
            "Live Detection of Dynamic Shapes (Webcam)",
        ),
    )

    if operation_type == "Upload Image for Dynamic Shape Classification":
        dynamicDetection()

    elif operation_type == "Live Detection of Dynamic Shapes (Webcam)":
        liveDynamic()

if __name__ == "__main__":
    main()
