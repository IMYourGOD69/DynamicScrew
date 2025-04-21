import streamlit as st
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from io import BytesIO

# Language translations
translations = {
    "en": {
        "title": "Dynamic Objects Counting Using Image Processing",
        "upload_image_label": "Upload an Image",
        "failed_load_image": "Failed to load image.",
        "image_caption": "Detection Dynamic Objects",
        "choose_operation": "Choose an Operation:",
        "select_option": "Select an option",
        "upload_image_classification": "Upload Image for Dynamic Shape Classification",
        "live_detection": "Live Detection of Dynamic Shapes (Webcam)",
        "webcam_title": "Live Dynamic Shape Detection (Webcam)",
        "enter_webcam_id": "Enter Webcam ID (usually 0 or 1):",
        "start_live": "Start Live Dynamic Shape Detection",
        "stop_live": "Stop Live Dynamic Detection",
        "webcam_failed": "Failed to open webcam. Please check the webcam ID.",
        "frame_failed": "Failed to capture frame from webcam.",
        "classification_title": "Dynamic Object Classification"
    },
    "zh": {
        "title": "使用图像处理的动态对象计数",
        "upload_image_label": "上传图像",
        "failed_load_image": "图像加载失败。",
        "image_caption": "检测动态对象",
        "choose_operation": "选择操作：",
        "select_option": "选择一个选项",
        "upload_image_classification": "上传图像进行动态形状分类",
        "live_detection": "实时检测动态形状（摄像头）",
        "webcam_title": "实时动态形状检测（摄像头）",
        "enter_webcam_id": "输入摄像头ID（通常为0或1）：",
        "start_live": "开始实时动态形状检测",
        "stop_live": "停止实时动态检测",
        "webcam_failed": "无法打开摄像头。请检查摄像头ID。",
        "frame_failed": "无法从摄像头捕获帧。",
        "classification_title": "动态对象分类"
    }
}

# Upload image with language support
def upload_image(upload_label):
    uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        return image
    return None

# Dynamic image processing
def process_dynamicImage(image, t):
    if image is None:
        st.error(t["failed_load_image"])
        return

    max_width = 800
    height, width = image.shape[:2]
    if width > max_width:
        scaling_factor = max_width / width
        image = cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 200)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                break
        else:
            size_groups.append((area, 1))

    colors = [tuple(random.sample(range(256), 3)) for _ in size_groups]
    output = image.copy()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 800:
                cv2.drawContours(output, [contour], -1, colors[idx], 2)
                M = cv2.moments(contour)
                if M["m00"]:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    cv2.putText(output, chr(65 + idx), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                break

    for i, (area, count) in enumerate(size_groups):
        mean_area = area / count
        cv2.putText(output, f"Size {chr(65+i)}: {count} (Mean Area: {mean_area:.1f})", (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[i], 2)

    st.subheader(t["classification_title"])
    st.pyplot(plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)), clear_figure=True)

def dynamicDetection(t):
    image = upload_image(t["upload_image_label"])
    process_dynamicImage(image, t)

stable_colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (128, 128, 0)
]

def process_liveDynamic(frame):
    max_width = 800
    height, width = frame.shape[:2]
    if width > max_width:
        frame = cv2.resize(frame, (int(width * max_width / width), int(height * max_width / width)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 50, 200)
    thresh = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    size_groups = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 500:
                size_groups[idx] = (group_area + area, count + 1)
                break
        else:
            size_groups.append((area, 1))

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        for idx, (group_area, count) in enumerate(size_groups):
            if abs(group_area / count - area) <= 800:
                color = stable_colors[idx % len(stable_colors)]
                cv2.drawContours(frame, [contour], -1, color, 2)
                break

    for i, (area, count) in enumerate(size_groups):
        mean_area = area / count
        cv2.putText(frame, f"Size {chr(65 + i)}: {count} (Mean Area: {mean_area:.1f})", (10, 50 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, stable_colors[i % len(stable_colors)], 3)
    return frame

def liveDynamic(t):
    st.subheader(t["webcam_title"])
    webcam_id = st.number_input(t["enter_webcam_id"], value=1, step=1)
    start = st.checkbox(t["start_live"])
    placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(webcam_id)
        if not cap.isOpened():
            st.error(t["webcam_failed"])
            return

        stop = False
        while not stop:
            ret, frame = cap.read()
            if not ret:
                st.warning(t["frame_failed"])
                break
            placeholder.image(process_liveDynamic(frame), channels="BGR")
            if st.button(t["stop_live"]):
                stop = True

        cap.release()
        cv2.destroyAllWindows()

def main():
    lang = st.sidebar.selectbox("Language / 语言", ("English", "Chinese"))
    t = translations[lang]

    st.title(t["title"])
    st.image("DynamicScrew.png", caption=t["image_caption"], use_container_width=True)

    operation = st.sidebar.selectbox(
        t["choose_operation"],
        (
            t["select_option"],
            t["upload_image_classification"],
            t["live_detection"]
        )
    )

    if operation == t["upload_image_classification"]:
        dynamicDetection(t)
    elif operation == t["live_detection"]:
        liveDynamic(t)

if __name__ == "__main__":
    main()
