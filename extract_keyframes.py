import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 추적할 팔과 손의 랜드마크 인덱스
ARM_HAND_LANDMARKS = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def get_pose_mask(frame, prev_mask):
    """
    MediaPipe Pose를 사용해 프레임에서 '팔과 손' 영역의 마스크를 생성합니다.
    감지되지 않으면 이전 마스크를 반환합니다.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    if results.pose_landmarks:
        h, w, _ = frame.shape
        landmarks = results.pose_landmarks.landmark
        
        found_landmarks = False
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for idx in ARM_HAND_LANDMARKS:
            if landmarks[idx].visibility > 0.3:
                found_landmarks = True
                x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
        
        if found_landmarks:
            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)
            return mask
        else:
            return prev_mask
    else:
        return prev_mask

def extract_keyframes(video_path, num_keyframes=10):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: {video_path}를 열 수 없습니다.")
        return None

    all_frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
    finally:
        cap.release()

    if len(all_frames) < num_keyframes:
        print("오류: 영상이 너무 짧거나 프레임이 부족합니다.")
        return None

    motion_scores = []
    prev_mask = get_pose_mask(all_frames[0], np.zeros(all_frames[0].shape[:2], dtype=np.uint8))
    prev_gray = cv2.cvtColor(all_frames[0], cv2.COLOR_BGR2GRAY)

    for frame in all_frames[1:]:
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_mask = get_pose_mask(frame, prev_mask)
        frame_diff = cv2.absdiff(prev_gray, current_gray)
        masked_diff = cv2.bitwise_and(frame_diff, frame_diff, mask=current_mask)
        thresh = cv2.threshold(masked_diff, 15, 255, cv2.THRESH_BINARY)[1]
        score = np.sum(thresh)
        motion_scores.append(score)
        prev_gray = current_gray
        prev_mask = current_mask

    if not motion_scores:
        print("오류: 움직임 점수 계산에 실패했습니다.")
        return None

    scores_np = np.array(motion_scores)
    threshold = max(np.mean(scores_np) + 0.5 * np.std(scores_np), 300) 
    motion_indices = np.where(scores_np > threshold)[0]

    if len(motion_indices) < num_keyframes:
        print(f"경고: 의미 있는 동작 구간을 찾지 못했습니다. (기준치: {threshold:.0f})")
        print("대체 로직: 전체 영상에서 균등 샘플링합니다.")
        key_indices = np.linspace(0, len(all_frames) - 1, num_keyframes, dtype=int)
    else:
        start_frame_idx = motion_indices[0] + 1 
        end_frame_idx = motion_indices[-1] + 1
        print(f"동작 구간 감지: 프레임 {start_frame_idx} ~ {end_frame_idx} (기준치: {threshold:.0f})")
        key_indices = np.linspace(start_frame_idx, end_frame_idx, num_keyframes, dtype=int)
        key_indices = np.clip(key_indices, 0, len(all_frames) - 1)

    keyframes = [all_frames[i] for i in key_indices]
    
    print(f"총 {len(keyframes)}개의 키프레임 추출 완료. (인덱스: {key_indices})")
    return keyframes