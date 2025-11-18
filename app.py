import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch


st.set_page_config(page_title="Cashew Leaf Health Analyzer", layout="centered")
st.title("🌿 ỨNG DỤNG PHÁT HIỆN VÀ PHÂN LOẠI BỆNH TRÊN LÁ ĐIỀU (YOLOv8)")

st.markdown("""
Ứng dụng sử dụng **2 mô hình YOLOv8**:
- 🩺 *Phân loại* (Classification): nhận dạng loại bệnh.
- 🖼️ *Khoanh vùng* (Detection): phát hiện và vẽ khung vùng bệnh.
""")


@st.cache_resource
def load_models():
    classify_path = "D:/cashew_yolo/ptud/runs/classify/cashew_classification/weights/best.pt"  
    detect_path = "D:/cashew_yolo/PTUD.v3i.yolov8/runs/detect/cashew_disease_detect/weights/best.pt"      

    try:
        classify_model = YOLO(classify_path)
    except Exception as e:
        st.error(f"❌ Không thể tải model phân loại: {e}")
        classify_model = None

    try:
        detect_model = YOLO(detect_path)
    except Exception as e:
        st.error(f"❌ Không thể tải model khoanh vùng: {e}")
        detect_model = None

    return classify_model, detect_model

classify_model, detect_model = load_models()
if classify_model and detect_model:
    st.success("✅ Cả hai mô hình đã được tải thành công!")

mode = st.selectbox(
    "Chọn chế độ hoạt động:",
    ["🩺 Phân loại bệnh", "🖼️ Khoanh vùng vùng bệnh"],
)


uploaded_file = st.file_uploader("📤 Tải lên ảnh lá điều", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh gốc", use_container_width=True)

    if mode == "🩺 Phân loại bệnh":
        if classify_model is None:
            st.warning("⚠️ Chưa có mô hình phân loại.")
        else:
            st.write("🔍 Đang phân loại...")
            results = classify_model.predict(image, imgsz=224, device=0 if torch.cuda.is_available() else "cpu")
            pred = results[0]
            cls_name = pred.names[pred.probs.top1]
            confidence = pred.probs.top1conf.item() * 100
            st.subheader(f"🩺 Kết quả: **{cls_name}** ({confidence:.2f}%)")

    elif mode == "🖼️ Khoanh vùng vùng bệnh":
        if detect_model is None:
            st.warning("⚠️ Chưa có mô hình khoanh vùng.")
        else:
            st.write("🔍 Đang phát hiện và khoanh vùng vùng bệnh...")
            results = detect_model.predict(image, conf=0.5, device=0 if torch.cuda.is_available() else "cpu")
            result_img = results[0].plot()
            st.image(result_img, caption="Ảnh đã khoanh vùng vùng bệnh", use_container_width=True)

else:
    st.info("⬆️ Hãy tải lên 1 ảnh để bắt đầu dự đoán.")

st.markdown("---")
st.caption("Phát triển bởi 🧠 Bạn • Mô hình: YOLOv8n • Framework: Streamlit 🚀")
