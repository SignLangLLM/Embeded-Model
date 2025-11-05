import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# SBERT 모델 로드 (프로그램 시작 시 1번만)
try:
    print("SBERT 모델 로드 중... (intfloat/multilingual-e5-large)")
    sbert_model = SentenceTransformer('intfloat/multilingual-e5-large')
    print("SBERT 모델 로드 성공.")
except Exception as e:
    print(f"SBERT 모델 로드 실패: {e}")
    sbert_model = None

def setup_db(sbert_model):
    """
    [임베딩 방식]
    수어 DB를 '단어: 정석 설명 문장'으로 구축하고,
    모든 문장을 임베딩하여 벡터로 반환합니다.
    """
    if sbert_model is None:
        return None, None, None
        
    print("KSL 정석 설명 DB 로드 및 임베딩 시작...")

    sign_language_db = {
        "가능": "손끝이 위로 향하게 펴서 세운 오른 손바닥을 입 앞에 댔다가 내밀며 입으로 ‘파’한다.",
        "고속도로": "오른 주먹의 1·5지를 펴서 끝이 밖으로 향하게 하여 빠르게 내밀며 1·5지 끝을 맞댄 다음, 손끝이 밖으로 향하게 모로 세운 두 손을 마주 보게 하여 밖으로 내민다.",
        "기본": "손등이 밖으로 향하게 주먹을 쥐고 세운 왼팔의 팔꿈치 밑에 오른 주먹의 등을 대고 손가락을 편다.",
        "끄다": "손끝이 약간 위로 손바닥이 밖으로 향하게 편 두 손의 5지에 1·2·3·4지를 모아 댄다.",
        "내리다": "손바닥이 위로 향하게 편 왼 손바닥에 오른 주먹의 1·2지를 펴서 끝을 댔다가 오른쪽 아래로 내린다.",
        "느리다": "손등이 위로 향하게 편 왼 손등에 오른손을 모로 세워 천천히 밖으로 내민다.",
        "도착": "왼손을 펴서 손바닥이 오른쪽으로 손끝이 밖으로 향하게 하고, 오른손을 약간 구부려 끝을 오른쪽에서부터 왼 손바닥에 가져다 대는 동작을 천천히 크게 한다.",
        "돈": "오른손의 1·5지 끝을 맞대어 동그라미를 만들어 약간 힘주어 내민다.",
        "되다": "오른쪽 어깨 앞에서 오른손을, 손바닥이 밖으로 향하게 세웠다가 손등이 밖으로 향하도록 손목을 돌린다.",
        "맞다": "오른 주먹의 1·5지를 펴서 끝이 이마로 향하게 하고, 왼 주먹의 1·5지를 펴서 끝이 턱으로 향하게 하였다가 두 손을 동시에 밖으로 내밀며 1·5지 끝을 맞댄다.",
        "물건": "손끝이 밖으로 향하게 펴서 모로 세운 왼 손바닥을 오른손의 5지 바닥으로 두 번 스쳐 낸다.",
        "받다": "손바닥이 위로 손끝이 밖으로 향하게 비스듬히 세운 두 손을 안으로 당긴다",
        "밤": "손바닥이 밖으로 향하게 세운 두 손을 중앙으로 모아 교차시킨다.",
        "분": "오른손 1지를 턱에 댔다가 약간 내린다.",
        "빨리": "오른 주먹의 1·5지를 펴서 끝이 위로 향하게 하여 위로 빠르게 올리며 끝을 맞댄다.",
        "숫자": "오른 주먹을 오른쪽 어깨 앞에서 왼쪽으로 이동시키면서 1·2·3·4지를 차례로 편다.",
        "승용차": "왼손을 약간 구부려 등이 위로 향하게 하고, 오른손을 펴서 왼손 밑에서 전후로 두 번 움직인다.",
        "얼마": "손바닥이 위로 향하게 편 오른손의 손가락을 쥐었다 폈다 한다.",
        "에어컨": "두 주먹을 가슴 앞으로 올려 떠는 동작을 한 다음, 1·2·5지를 약간 구부린 두 주먹을 모로 세워 오른 주먹을 왼 주먹 약간 뒤에 놓고 동시에 상하로 흔든다.",
        "여기": "오른 주먹의 1지를 펴서 끝이 약간 아래로 향하게 하여 약간 내린다.",
        "열다": "5지를 접고 손바닥이 밖으로 향하게 세운 두 손을 맞댔다가 좌우로 벌려 두 손바닥이 마주보게 한다.",
        "옮기다": "오른손을 구부려 손바닥이 위로 향하게 하여 왼쪽에서 오른쪽으로 움직인다.",
        "우회전": "손등이 위로 손끝이 밖으로 향하게 약간 구부린 오른손을 밖으로 내밀다가 오른쪽으로 돌린다.",
        "운전": "가볍게 쥔 두 주먹을 가슴 앞으로 올려 좌우로 두 번 돌린다.",
        "은행": "두 주먹의 손목을 상하로 두 번 마주 댔다 뗀다.",
        "이후": "손등이 밖으로 손끝이 오른쪽으로 향하게 편 왼 손등에, 손바닥이 밖으로 손끝이 위로 향하게 편 오른 손등을 댔다가 밖으로 내민다.",
        "좌회전": "손등이 위로 손끝이 밖으로 향하게 약간 구부린 오른손을 밖으로 내밀다가 왼쪽으로 돌린다.",
        "주다": "오른손을 펴서 손바닥이 위로 손끝이 밖으로 향하게 하여 밖으로 내민다.",
        "지금": "1·2·5지를 펴서 등이 위로 끝이 밖으로 향하게 한 두 손을 아래로 약간 내린다.",
        "지름길": "오른 주먹의 등을 입 앞에 대고 1·2지와 5지를 댔다 뗀 다음, 손끝이 밖으로 향하게 모로 세운 두 손을 마주 보게 하여 좌우로 움직이며 밖으로 내민다.",
        "추가": "1·5지를 펴서 1지가 위에 5지가 아래에 놓이게 하여 약간 구부린 오른 주먹을, 같은 모양의 왼 주먹의 5지 밑에 댔다가 왼 주먹의 1지 위에 올려놓는다.",
        "카드": "오른 주먹의 1·5지를 펴서 약간 구부려 끝이 밖으로 향하게 하여 위아래로 흔든다.",
        "택시": "오른 주먹의 5지를 펴서 1지 끝 바닥에 대고 세우고 전후로 흔든다.",
        "트렁크": "오른 주먹의 1·5지를 펴서 약간 구부려 끝이 밖으로 향하게 하여 아래로 내린다."
    }

    target_labels = [
        "가능", "고속도로", "기본", "끄다", "내리다", "느리다", "도착", "돈", "되다", "맞다",
        "물건", "받다", "밤", "분", "빨리", "숫자", "승용차", "얼마", "에어컨", "여기",
        "열다", "옮기다", "우회전", "운전", "은행", "이후", "좌회전", "주다", "지금", "지름길",
        "추가", "카드", "택시", "트렁K크"
    ]
    filtered_db = {label: desc for label, desc in sign_language_db.items() if label in target_labels}
    
    if len(filtered_db) != 34:
        print(f"경고: DB 단어가 34개가 아닙니다. (현재: {len(filtered_db)}개)")
        
    db_labels = list(filtered_db.keys())
    db_sentences = list(filtered_db.values())

    try:
        db_embeddings = sbert_model.encode(db_sentences)
        print(f"수어 DB 임베딩 완료 ({len(db_labels)}개 단어).")
        return db_labels, db_embeddings, filtered_db
    except Exception as e:
        print(f"DB 임베딩 실패: {e}")
        return None, None, None

def find_best_match(llm_description, sbert_model, db_labels, db_embeddings):
    if sbert_model is None or llm_description is None:
        return None, 0

    try:
        query_embedding = sbert_model.encode([llm_description])
        similarities = cosine_similarity(query_embedding, db_embeddings)

        best_match_index = np.argmax(similarities[0])
        best_score = similarities[0][best_match_index]
        best_label = db_labels[best_match_index]

        return best_label, best_score
    
    except Exception as e:
        print(f"유사도 비교 실패: {e}")
        return None, 0