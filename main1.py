import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow.lite as tflite
import pygame
import math
import time

# === Cấu hình ban đầu ===
# Đường dẫn file mô hình và âm thanh
MODEL_PATH = os.path.join("assets", "face_landmarker.task")  # Mô hình Mediapipe để phát hiện khuôn mặt
EYE_MODEL_PATH = os.path.join("assets", "eye_classifier_ver3.tflite")  # Mô hình TFLite để phân loại mắt
ALARM_PATH = os.path.join("assets", "alarm.mp3")  # File âm thanh cảnh báo
OUTPUT_DIR = "eye_images"  # Thư mục lưu ảnh mắt tạm thời

# Kiểm tra file tồn tại
for path, name in [(MODEL_PATH, "face_landmarker.task"), (EYE_MODEL_PATH, "eye_classifier_ver3.tflite"), (ALARM_PATH, "alarm.mp3")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} không tìm thấy tại: {path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)  # Tạo thư mục lưu ảnh tạm thời nếu chưa có

# Khởi tạo Mediapipe Face Landmarker (phát hiện khuôn mặt và landmark)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1  # Chỉ phát hiện 1 khuôn mặt để tối ưu tài nguyên
    )
)

# Khởi tạo mô hình TFLite để phân loại trạng thái mắt (open/closed)
interpreter = tflite.Interpreter(model_path=EYE_MODEL_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()

# Khởi tạo pygame để phát âm thanh cảnh báo
pygame.mixer.init()
pygame.mixer.set_num_channels(10)  # Tăng số kênh âm thanh để hỗ trợ chồng chập
ALARM_SOUND = pygame.mixer.Sound(ALARM_PATH)

# Khởi tạo webcam và điều chỉnh độ phân giải thành hình vuông
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ValueError("Không thể mở webcam")
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resolution = min(width, height)  # Lấy giá trị nhỏ hơn để tạo khung hình vuông
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution)

# === Cấu hình trạng thái và ngưỡng ===
STATE = {  # Trạng thái của hệ thống
    "frame_count": 0,  # Đếm số khung hình đã xử lý
    "closed_eye_frame_count": 0,  # Đếm số khung hình mà cả hai mắt nhắm
    "left_eye_closed_count": 0,  # Đếm số khung hình mà mắt trái nhắm
    "right_eye_closed_count": 0,  # Đếm số khung hình mà mắt phải nhắm
    "is_drowsy": False,  # Trạng thái buồn ngủ (True/False)
    "last_left_eye_status": "open",  # Trạng thái mắt trái gần nhất
    "last_right_eye_status": "open",  # Trạng thái mắt phải gần nhất
    "last_alarm_time": 0  # Thời gian phát âm thanh cảnh báo cuối cùng
}
CONFIG = {  # Các ngưỡng và tham số cấu hình
    "drowsiness_threshold_both": 7,  # Số khung hình nhắm cả hai mắt để coi là buồn ngủ
    "drowsiness_threshold_one": 12,  # Số khung hình nhắm một mắt để coi là buồn ngủ
    "face_turn_threshold_left": 40,  # Ngưỡng góc nghiêng để phát hiện sideview bên trái
    "face_turn_threshold_right": 40,  # Ngưỡng góc nghiêng để phát hiện sideview bên phải
    "max_sideview_threshold": 70,  # Ngưỡng góc nghiêng tối đa để bỏ qua khung hình
    "left_eye_indices": [133, 33, 159, 145],  # Chỉ số landmark cho mắt trái
    "right_eye_indices": [263, 362, 385, 380],  # Chỉ số landmark cho mắt phải
    "padding": 5,  # Độ mở rộng vùng cắt mắt (pixel)
    "eye_size": 128,  # Kích thước ảnh mắt (128x128)
    "alarm_interval": 0.8  # Khoảng thời gian giữa các lần phát âm thanh (giây)
}

# === Các hàm hỗ trợ ===
def calculate_turn_angle(landmarks):
    """Tính góc nghiêng của khuôn mặt dựa trên các điểm landmark."""
    nose, left_eye, right_eye = landmarks[1], landmarks[33], landmarks[263]
    nose_3d = np.array([nose.x, nose.y, nose.z])
    left_eye_3d = np.array([left_eye.x, left_eye.y, left_eye.z])
    right_eye_3d = np.array([right_eye.x, right_eye.y, right_eye.z])
    face_direction = (left_eye_3d - nose_3d + right_eye_3d - nose_3d) / 2
    return math.degrees(math.atan2(face_direction[0], face_direction[2]))

def get_eye_rect(landmarks, indices, frame_shape):
    """Tính vùng bao quanh mắt (hình vuông) từ các điểm landmark."""
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    for idx in indices:
        lm = landmarks[idx]
        x, y = lm.x * frame_shape[1], lm.y * frame_shape[0]
        min_x, max_x = min(min_x, x), max(max_x, x)
        min_y, max_y = min(min_y, y), max(max_y, y)
    max_side = max(max_x - min_x, max_y - min_y)
    center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    half_side = (max_side / 2) + CONFIG["padding"]
    return (
        int(max(0, center_x - half_side)),
        int(max(0, center_y - half_side)),
        int(min(frame_shape[1], center_x + half_side)),
        int(min(frame_shape[0], center_y + half_side))
    )

def classify_eye(image):
    """Phân loại trạng thái mắt (open/closed) bằng mô hình TFLite."""
    if image is None or image.size == 0:
        return "unknown"
    input_data = image.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=(0, -1))
    interpreter.set_tensor(INPUT_DETAILS[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])
    return "closed" if output_data[0][1] > output_data[0][0] else "open"

def process_eye(frame, rect, path, status_key):
    """Xử lý và phân loại trạng thái của một mắt (trái/phải)."""
    if rect is None:
        return "unknown"
    img = frame[rect[1]:rect[3], rect[0]:rect[2]]
    if img.size == 0:
        return "unknown"
    img = cv2.resize(img, (CONFIG["eye_size"], CONFIG["eye_size"]), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path, img)
    status = classify_eye(img)
    os.remove(path)
    STATE[status_key] = status
    return status

def reset_drowsiness_state():
    """Đặt lại trạng thái buồn ngủ khi không phát hiện khuôn mặt hoặc góc nghiêng lớn."""
    STATE.update({"left_eye_closed_count": 0, "right_eye_closed_count": 0, "closed_eye_frame_count": 0})
    if STATE["is_drowsy"]:
        STATE["is_drowsy"] = False
        pygame.mixer.stop()

# === Vòng lặp chính ===
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc khung hình từ webcam")
            break

        # Cắt khung hình thành hình vuông
        h, w = frame.shape[:2]
        if w != h:
            size = min(w, h)
            start_x, start_y = (w - size) // 2, (h - size) // 2
            frame = frame[start_y:start_y + size, start_x:start_x + size]

        frame = cv2.flip(frame, 1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_img, int(cap.get(cv2.CAP_PROP_POS_MSEC)))

        left_status = STATE["last_left_eye_status"] if not result.face_landmarks else "unknown"
        right_status = STATE["last_right_eye_status"] if not result.face_landmarks else "unknown"
        is_sideview = False
        turn_angle = 0

        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                turn_angle = calculate_turn_angle(landmarks)

                # Bỏ qua khung hình nếu góc nghiêng quá lớn
                if abs(turn_angle) > CONFIG["max_sideview_threshold"]:
                    reset_drowsiness_state()
                    continue

                # Kiểm tra trạng thái sideview
                is_sideview = turn_angle > CONFIG["face_turn_threshold_right"] if turn_angle > 0 else abs(turn_angle) > CONFIG["face_turn_threshold_left"]

                # Xác định vùng mắt cần xử lý
                left_rect = get_eye_rect(landmarks, CONFIG["left_eye_indices"], frame.shape) if not (is_sideview and turn_angle > 0) else None
                right_rect = get_eye_rect(landmarks, CONFIG["right_eye_indices"], frame.shape) if not (is_sideview and turn_angle < 0) else None

                # Phân loại trạng thái mắt
                left_status = process_eye(frame, left_rect, os.path.join(OUTPUT_DIR, f"left_eye_{STATE['frame_count']}.png"), "last_left_eye_status")
                right_status = process_eye(frame, right_rect, os.path.join(OUTPUT_DIR, f"right_eye_{STATE['frame_count']}.png"), "last_right_eye_status")

                # Cập nhật trạng thái buồn ngủ
                if is_sideview:
                    if turn_angle > 0:
                        STATE["right_eye_closed_count"] = STATE["right_eye_closed_count"] + 1 if right_status == "closed" else 0
                        STATE["left_eye_closed_count"] = 0
                    else:
                        STATE["left_eye_closed_count"] = STATE["left_eye_closed_count"] + 1 if left_status == "closed" else 0
                        STATE["right_eye_closed_count"] = 0
                    STATE["closed_eye_frame_count"] = 0
                else:
                    STATE["left_eye_closed_count"] = STATE["left_eye_closed_count"] + 1 if left_status == "closed" else 0
                    STATE["right_eye_closed_count"] = STATE["right_eye_closed_count"] + 1 if right_status == "closed" else 0
                    STATE["closed_eye_frame_count"] = STATE["closed_eye_frame_count"] + 1 if left_status == "closed" and right_status == "closed" else 0

                # Kiểm tra trạng thái buồn ngủ
                is_drowsy = (STATE["closed_eye_frame_count"] >= CONFIG["drowsiness_threshold_both"] or 
                             STATE["left_eye_closed_count"] >= CONFIG["drowsiness_threshold_one"] or 
                             STATE["right_eye_closed_count"] >= CONFIG["drowsiness_threshold_one"])
                
                if is_drowsy and not STATE["is_drowsy"]:
                    STATE["is_drowsy"] = True
                elif STATE["is_drowsy"] and (left_status == "open" and right_status == "open"):
                    STATE.update({"closed_eye_frame_count": 0, "left_eye_closed_count": 0, "right_eye_closed_count": 0, "is_drowsy": False})
                    pygame.mixer.stop()

                # Phát âm thanh cảnh báo dồn dập khi buồn ngủ
                if STATE["is_drowsy"]:
                    current_time = time.time()
                    if current_time - STATE["last_alarm_time"] >= CONFIG["alarm_interval"]:
                        ALARM_SOUND.play(0)  # Phát âm thanh một lần, cho phép chồng chập
                        STATE["last_alarm_time"] = current_time

                # Hiển thị thông tin trên khung hình
                cv2.putText(frame, f"Left Eye: {left_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Right Eye: {right_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Sideview: {is_sideview}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Turn Angle: {turn_angle:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                if is_drowsy:
                    cv2.putText(frame, "Drowsiness Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Vẽ các điểm landmark lên khung hình
                for lm in landmarks:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        else:
            reset_drowsiness_state()

        cv2.imshow("DrowsinessDetection", frame)
        STATE["frame_count"] += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Lỗi xảy ra: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    pygame.mixer.quit()
