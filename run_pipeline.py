import os
import cv2
import google.generativeai as genai
from PIL import Image 
import sys
import warnings
import glob 


try:
    # (1) SBERT (PyTorch) 임포트
    from embed_and_compare import setup_db, find_best_match, sbert_model 
    
    # (2) MediaPipe (TensorFlow) 임포트
    from extract_keyframes import extract_keyframes
except ImportError as e: 
    print("--- [디버깅] 임포트 실패 ---")
    print(f"발생한 실제 오류: {e}")
    print("----------------------------")
    print("오류: 'extract_keyframes.py' 또는 'embed_and_compare.py'를 임포트하는 중 문제가 발생했습니다.")
    print("라이브러리가 모두 설치되었는지 확인하세요. (pip install mediapipe sentence-transformers scikit-learn numpy)")
    sys.exit()


GOOGLE_API_KEY = "여기에입력" # ◀◀ 여기에 키 입력

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"API 키 설정 오류: {e}")
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
         print("오류: GOOGLE_API_KEY를 코드에 직접 입력해주세요.")
    sys.exit()

def analyze_frames_with_llm(keyframes):
    """
    [LLM 1: Vision]
    (기술적 묘사'를 유도하는 프롬프트)
    """
    print("\n--- 3단계 (LLM: KSL 분석) 이미지 분석 시작 ---")
    
    model = genai.GenerativeModel('gemini-2.5-flash') 
    image_parts = []
    for frame in keyframes:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        image_parts.append(pil_image)

    prompt_parts = [
        "당신은 KSL(한국수어)의 4요소(수형, 수위, 수동, 수향)를 분석하는 '기술 분석가'입니다.",
        "제공된 5장의 시퀀스 이미지는 동작의 [시작-중간-끝]을 보여줍니다.",
        "이 5장의 흐름을 보고, 동작을 '기술적인 KSL 용어'로만 분석하여 한 문장으로 묘사하세요.",
        
        "\n[중요 지침]",
        "1. '...같습니다', '...하는 것 같아요', '...처럼 보입니다' 같은 추측성, 감성적 묘사를 **절대** 사용하지 마세요.",
        "2. 손가락은 '엄지, 검지' 대신 '1지, 2지, 5지' 등의 숫자로 표현하세요.",
        "3. 방향은 '손바닥이 밖으로', '손등이 위로', '모로 세워' 등으로 명확히 표현하세요.",
        "4. 움직임은 '두 번 움직인다', '좌우로 벌린다', '오른쪽으로 돌린다' 등으로 객관적으로 서술하세요.",

        "\n[출력 예시 (이 형식과 동일하게 묘사하세요)]",
        "---",
        "예시 1: 5지를 접고 손바닥이 밖으로 향하게 세운 두 손을 맞댔다가 좌우로 벌려 두 손바닥이 마주보게 한다.",
        "예시 2: 가볍게 쥔 두 주먹을 가슴 앞으로 올려 좌우로 두 번 돌린다.",
        "예시 3: 오른 주먹의 1·5지를 펴서 약간 구부려 끝이 밖으로 향하게 하여 위아래로 흔든다.",
        "예시 4: 오른손 1지를 턱에 댔다가 약간 내린다.",
        "---",
        
        "\n[분석 시작]",
        "위 예시 형식을 완벽하게 따라서, 아래 5개 프레임의 동작을 '기술적인 한 문장'으로 묘사하세요:",
        
        *image_parts 
    ]

    try:
        response = model.generate_content(prompt_parts)
        print("LLM (KSL 분석) 응답 수신 완료.")
        return response.text
    except Exception as e:
        print(f"LLM (KSL 분석) API 호출 오류: {e}")
        return None

def process_single_video(video_path, sbert_model, db_labels, db_embeddings, db_full_text):
    """
    한 개의 비디오 파일에 대해 'SBERT 임베딩' 파이프라인을 실행합니다.
    """
    print(f"\n{'='*20} PROCESSING START {'='*20}")
    print(f"▶▶▶ 영상 파일: {os.path.basename(video_path)} ◀◀◀")
    
    # --- 1단계: 영상에서 키프레임 추출 ---
    print("\n--- 1단계: 키프레임 추출 시작 ---")
    keyframes = extract_keyframes(video_path, num_keyframes=5)
    
    if not keyframes or len(keyframes) < 1:
        print(f"파이프라인 중단: '{video_path}'에서 키프레임을 추출하지 못했습니다.")
        return

    # --- [추가 기능] 키프레임 파일로 저장 (한글 경로 수정됨) ---
    print("--- [추가] 키프레임 파일로 저장 시작 ---")
    output_dir = "output_frames"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"'{output_dir}' 폴더 생성 완료.")

    base_video_name = os.path.splitext(os.path.basename(video_path))[0]
    for i, frame in enumerate(keyframes):
        save_path = os.path.join(output_dir, f"{base_video_name}_keyframe_{i+1}.png")
        try:
            is_success, img_buffer = cv2.imencode(".png", frame)
            if is_success:
                with open(save_path, 'wb') as f:
                    f.write(img_buffer.tobytes())
            else:
                raise Exception("cv2.imencode failed")
        except Exception as e:
            print(f"'{save_path}' 저장 실패: {e}")
    print(f"키프레임 {len(keyframes)}개를 '{output_dir}' 폴더에 저장 완료.")
    # --- [저장 완료] ---

    # --- 3단계 (LLM): 이미지 -> '기술적' 텍스트 묘사 ---
    visual_description = analyze_frames_with_llm(keyframes) 
    
    if not visual_description:
        print("파이프라인 중단: LLM 1이 동작을 묘사하지 못했습니다.")
        return

    print("\n[LLM (KSL 분석) 묘사 결과]")
    print(f"-> {visual_description.strip()}")

    # --- 4 & 5단계 (SBERT): 묘사 vs DB -> 코사인 유사도 비교 ---
    print("\n--- 4/5단계 (SBERT) 임베딩 비교 시작 ---")
    best_label, best_score = find_best_match(
        visual_description, 
        sbert_model, 
        db_labels, 
        db_embeddings
    )

    if best_label:
        print("\n====================")
        print(f"  [{os.path.basename(video_path)}] 최종 수어 추론 결과")
        print("====================")
        print(f"  ->  '{best_label}'  (유사도: {best_score:.4f})")
        print(f"\n(참고: LLM 묘사) '{visual_description.strip()}'")
        print(f"(참고: DB 정답) '{db_full_text.get(best_label, 'N/A')}'")
    else:
        print("최종 추론에 실패했습니다. (SBERT 매칭 실패)")
    
    print(f"{'='*20} PROCESSING END {'='*20}\n")


# --- 이 스크립트를 실행 ---
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    VIDEO_FOLDER = "input_videos"
    
    if not os.path.exists(VIDEO_FOLDER):
        print(f"오류: '{VIDEO_FOLDER}' 폴더를 찾을 수 없습니다.")
        sys.exit()

    # --- 2단계 (준비): SBERT 모델 및 DB 임베딩 (★최초 1회만 실행★)
    if sbert_model is None:
        print("파이프라인 중단: SBERT 모델 로드에 실패했습니다.")
        sys.exit()
        
    db_labels, db_embeddings, db_full_text = setup_db(sbert_model) 
    if db_labels is None:
        print("파이프라인 중단: 수어 DB 로드 및 임베딩에 실패했습니다.")
        sys.exit()

    video_extensions = ["*.mp4", "*.mov", "*,avi", "*.mkv", "*.MP4", "*.MOV", "*.AVI", "*.MKV"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(VIDEO_FOLDER, ext)))

    if not video_files:
        print(f"오류: '{VIDEO_FOLDER}' 폴더에 처리할 영상 파일이 없습니다.")
        sys.exit()

    print(f"\n총 {len(video_files)}개의 비디오 파일을 찾았습니다. 파이프라인을 시작합니다.")
    
    for video_path in video_files:
        try:
            process_single_video(video_path, sbert_model, db_labels, db_embeddings, db_full_text)
        except Exception as e:
            print(f"!!!!!!!!! {os.path.basename(video_path)} 처리 중 치명적인 오류 발생: {e} !!!!!!!!!")