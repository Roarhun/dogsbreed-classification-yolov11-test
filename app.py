import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# โหลดโมเดล
model = YOLO("best.pt")

st.title("🐶 Dog Breed Classification Web App")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพสุนัข", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 รูปที่อัปโหลด', use_container_width=True)

    # ปุ่มให้ผู้ใช้กดเพื่อวิเคราะห์
    if st.button("🔍 วิเคราะห์"):
        # Convert image to NumPy array (for model)
        image_np = np.array(image)

        # YOLO ต้องการ BGR (เหมือน OpenCV)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # รันโมเดล
        results = model(image_bgr)

        # แสดงผลภาพพร้อมกล่องตรวจจับ
        res_plotted = results[0].plot()
        st.image(res_plotted, caption="🎯 ผลการจำแนก", use_container_width=True)


        # แสดง label ที่ตรวจพบ
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"🦴 ตรวจพบ: **{label}** ({conf:.2%})")
