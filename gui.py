import math
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO 

MODEL_PATH = r'D:\UEH\2 - HKĐ2026\AI\BUỔI 4\PALM\runs\detect\palm_finetune_fate\weights\best.pt'

print(f"🔄 Đang tải model YOLO từ {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Tải model thành công!")
except Exception as e:
    print(f"❌ Lỗi tải model, kiểm tra lại đường dẫn hoặc thư viện: {e}")
    exit()

PALMISTRY_LABELS = ["life", "heart", "head", "fate"]

def extract_lines(result):
    """Trích xuất các đường chỉ tay từ YOLO, bỏ qua mọi nhãn không thuộc palmistry."""
    names = result.names
    boxes = result.boxes
    detected = {}

    if boxes is None:
        return detected

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])

        if label not in PALMISTRY_LABELS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        length = math.sqrt(width ** 2 + height ** 2)

        line_info = {
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "length": length,
            "center_x": (x1 + x2) / 2,
            "center_y": (y1 + y2) / 2,
        }

        # Nếu cùng một đường bị detect nhiều lần, giữ box tự tin nhất.
        if label not in detected or conf > detected[label]["conf"]:
            detected[label] = line_info

    return detected

def classify_length(line_info, image_width, image_height):
    if line_info is None:
        return "missing"

    diag = math.sqrt(image_width ** 2 + image_height ** 2)
    ratio = line_info["length"] / diag if diag else 0

    if ratio >= 0.45:
        return "long"
    if ratio >= 0.28:
        return "medium"
    return "short"

def classify_slope(line_info):
    if line_info is None:
        return "missing"

    dx = line_info["x2"] - line_info["x1"]
    decay_y = line_info["y2"] - line_info["y1"]

    if abs(dx) < 1:
        return "vertical"

    slope = decay_y / dx
    if slope > 0.4:
        return "downward"
    if slope < -0.4:
        return "upward"
    return "flat"

def get_palmistry_reading(lines, image_width, image_height):
    reading = []

    if not lines:
        reading.append("Không phát hiện rõ đường chỉ tay nào.")
        reading.append("Hãy căn chỉnh tay thẳng và rõ nét hơn nhé.")
        return reading

    life_line = lines.get("life")
    heart_line = lines.get("heart")
    head_line = lines.get("head")

    life_len = classify_length(life_line, image_width, image_height)
    heart_len = classify_length(heart_line, image_width, image_height)
    head_len = classify_length(head_line, image_width, image_height)
    fate_len = classify_length(lines.get("fate"), image_width, image_height)
    head_slope = classify_slope(head_line)

    if life_len == "long":
        reading.append("+ Đ.Sinh mệnh dài: Bền bỉ, sức chịu đựng tốt.")
    elif life_len == "medium":
        reading.append("+ Đ.Sinh mệnh vừa: Năng lượng cân bằng, dễ thích nghi.")
    elif life_len == "short":
        reading.append("+ Đ.Sinh mệnh ngắn: Thiên về thay đổi, nên nghỉ ngơi.")
    else:
        reading.append("+ Không rõ đường Sinh mệnh.")

    if heart_len == "long":
        reading.append("+ Đ.Tình cảm dài: Sống tình cảm, gắn bó sâu sắc.")
    elif heart_len == "medium":
        reading.append("+ Đ.Tình cảm vừa: Cân bằng cảm xúc & lý trí.")
    elif heart_len == "short":
        reading.append("+ Đ.Tình cảm ngắn: Kín đáo, ít bộc lộ cảm xúc.")
    else:
        reading.append("+ Không rõ đường Tình cảm.")

    if head_len == "long":
        reading.append("+ Đ.Trí đạo dài: Suy nghĩ sâu, phân tích kỹ.")
    elif head_len == "medium":
        reading.append("+ Đ.Trí đạo vừa: Tư duy linh hoạt, thực tế.")
    elif head_len == "short":
        reading.append("+ Đ.Trí đạo ngắn: Hành động nhanh, quyết định trực giác.")
    else:
        reading.append("+ Không rõ đường Trí đạo.")

    if head_slope == "downward":
        reading.append("+ Trí đạo xuôi: Giàu tưởng tượng, sáng tạo.")
    elif head_slope == "flat":
        reading.append("+ Trí đạo ngang: Logic, thực tế và ổn định.")
    elif head_slope == "upward":
        reading.append("+ Trí đạo hướng lên: Tham vọng, mục tiêu rõ ràng.")

    if fate_len == "long":
        reading.append("+ Đ.Vận mệnh dài: Sự nghiệp rõ ràng, kiên trì.")
    elif fate_len == "medium":
        reading.append("+ Đ.Vận mệnh vừa: Sự nghiệp thay đổi theo giai đoạn.")
    elif fate_len == "short":
        reading.append("+ Đ.Vận mệnh mờ: Tự tạo cơ hội, lộ trình tự do.")
    else:
        reading.append("+ Không rõ đường Vận mệnh.")

    missing = [label for label in PALMISTRY_LABELS if lines.get(label) is None]
    if missing:
        reading.append("+ Gợi ý: Căn tay sáng/rõ hơn để thấy thêm: " + ", ".join(missing) + ".")

    return reading

class PalmistryGUI:
    def __init__(self, root, model_yolo):
        self.root = root
        self.model = model_yolo

        self.root.title("Palmistry AI System")
        self.root.geometry("1100x650")

        self.cap = None
        self.current_frame = None
        self.is_webcam_running = False
        self.is_frozen = False

        self.left_panel = tk.Label(root, bg="black")
        self.left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right_panel = tk.Frame(root, width=360, bg="#111111")
        self.right_panel.pack(side="right", fill="y", padx=10, pady=10)
        self.right_panel.pack_propagate(False)

        title = tk.Label(
            self.right_panel,
            text="KẾT QUẢ XEM CHỈ TAY",
            font=("Segoe UI", 16, "bold"),
            fg="#00ffff",
            bg="#111111"
        )
        title.pack(pady=(15, 10))

        self.result_text = tk.Text(
            self.right_panel,
            font=("Segoe UI", 12),
            fg="white",
            bg="#111111",
            wrap="word",
            borderwidth=0,
            height=22
        )
        self.result_text.pack(fill="both", expand=True, padx=15)

        self.btn_webcam = tk.Button(
            self.right_panel,
            text="Mở webcam realtime",
            font=("Segoe UI", 11),
            command=self.start_webcam
        )
        self.btn_webcam.pack(fill="x", padx=15, pady=5)

        self.btn_import = tk.Button(
            self.right_panel,
            text="Import ảnh",
            font=("Segoe UI", 11),
            command=self.import_image
        )
        self.btn_import.pack(fill="x", padx=15, pady=5)

        self.btn_restart = tk.Button(
            self.right_panel,
            text="R - Restart / quét lại",
            font=("Segoe UI", 11),
            command=self.reset_webcam
        )
        self.btn_restart.pack(fill="x", padx=15, pady=5)

        self.btn_quit = tk.Button(
            self.right_panel,
            text="Q - Thoát",
            font=("Segoe UI", 11),
            command=self.close
        )
        self.btn_quit.pack(fill="x", padx=15, pady=(5, 15))

        # Đăng ký phím tắt hệ thống
        self.root.bind("<c>", lambda event: self.capture_webcam_frame())
        self.root.bind("<C>", lambda event: self.capture_webcam_frame())
        self.root.bind("<r>", lambda event: self.reset_webcam())
        self.root.bind("<R>", lambda event: self.reset_webcam())
        self.root.bind("<q>", lambda event: self.close())
        self.root.bind("<Q>", lambda event: self.close())
        self.root.bind("<Escape>", lambda event: self.close())

        self.show_message([
            "HỆ THỐNG XEM CHỈ TAY AI",
            "Chọn một chế độ hành động:",
            "- Mở webcam realtime để quét trực tiếp lòng bàn tay.",
            "- Nhấn C trên bàn phím để chụp/chốt kết quả phân tích.",
            "- Nhấn R để khởi động lại luồng quét.",
            "- Bấm Import ảnh để phân tích chỉ tay từ file ảnh có sẵn."
        ])

    def show_message(self, lines):
        self.result_text.delete("1.0", tk.END)
        for line in lines:
            self.result_text.insert(tk.END, line + "\n\n")

    def predict_frame(self, frame):
        image_height, image_width = frame.shape[:2]

        # Dự đoán qua YOLO
        results = self.model.predict(
            source=frame,
            conf=0.25,
            save=False,
            verbose=False
        )

        result = results[0]
        annotated_frame = result.plot()

        lines = extract_lines(result)
        reading = get_palmistry_reading(lines, image_width, image_height)

        return annotated_frame, reading

    def show_frame_on_gui(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((720, 600))

        imgtk = ImageTk.PhotoImage(image=img)
        self.left_panel.imgtk = imgtk
        self.left_panel.configure(image=imgtk)

    def start_webcam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.show_message(["LỖI: Không thể kết nối thiết bị Webcam."])
            return

        self.is_webcam_running = True
        self.is_frozen = False

        self.show_message([
            "🎥 CHẾ ĐỘ WEBCAM TRỰC TIẾP",
            "Webcam đang quét thời gian thực.",
            "Hãy hướng lòng bàn tay đối diện camera.",
            "Nhấn C để chụp/chốt kết quả bói chỉ tay."
        ])
        self.update_webcam()

    def update_webcam(self):
        if not self.is_webcam_running or self.is_frozen:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.show_message(["LỖI: Mất tín hiệu truyền nhận khung hình luồng."])
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()

        annotated_frame, _ = self.predict_frame(frame)

        cv2.putText(
            annotated_frame,
            "LIVE - C: CAPTURE | R: RESTART | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        self.show_frame_on_gui(annotated_frame)
        self.root.after(30, self.update_webcam)

    def capture_webcam_frame(self):
        if self.current_frame is None:
            self.show_message(["Chưa có dữ liệu từ camera để phân tích chụp."])
            return

        self.is_frozen = True
        self.is_webcam_running = False

        annotated_frame, reading = self.predict_frame(self.current_frame)

        cv2.putText(
            annotated_frame,
            "CAPTURED - R: RESTART | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        self.show_frame_on_gui(annotated_frame)
        self.show_message(reading)

    def reset_webcam(self):
        self.is_frozen = False
        self.is_webcam_running = True

        self.show_message([
            "🔄 KHỞI ĐỘNG LẠI THÀNH CÔNG",
            "Căn tay thẳng, rõ nét trong khung camera.",
            "Nhấn C để chụp chốt và phân tích kết quả."
        ])
        self.update_webcam()

    def import_image(self):
        self.is_webcam_running = False
        self.is_frozen = True

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh bàn tay phân tích chỉ tay",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            with open(file_path, "rb") as f:
                chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            frame = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            frame = None
            print(f"Lỗi đọc file: {e}")

        if frame is None:
            self.show_message(["LỖI: Hệ thống không thể giải mã hình ảnh này."])
            return

        self.current_frame = frame.copy()
        annotated_frame, reading = self.predict_frame(frame)

        cv2.putText(
            annotated_frame,
            "IMAGE MODE - R: WEBCAM | Q: QUIT",
            (20, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        self.show_frame_on_gui(annotated_frame)
        self.show_message(reading)

    def close(self):
        self.is_webcam_running = False
        self.is_frozen = True

        if self.cap is not None:
            self.cap.release()

        self.root.destroy()
        print("👋 Đã tắt ứng dụng Palmistry AI thành công.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PalmistryGUI(root, model)
    root.mainloop()