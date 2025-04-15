import os
import openai
import deepl
from anthropic import Client
import re
import time

# Gemini
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    gemini_enabled = True
except ImportError:
    gemini_enabled = False
    print("[Warning] google-generativeai 패키지가 설치되지 않았습니다. Gemini 사용 불가.")

###############################################################################
# 1) 환경 변수로 API KEY 읽기
###############################################################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")

# 설정
openai.api_key = OPENAI_API_KEY
translator_deepl = deepl.Translator(DEEPL_API_KEY) if DEEPL_API_KEY else None
anthropic_client = Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

###############################################################################
# 2) 모델 정보 사전
###############################################################################
MODEL_INFO = {
    "openai_gpt_4o": {"type": "openai", "model_name": "gpt-4o"},
    "openai_gpt_4o_mini": {"type": "openai", "model_name": "gpt-4o-mini"},
    "openai_gpt_o3_mini": {"type": "openai", "model_name": "o3-mini"},

    "claude_sonnet": {"type": "anthropic", "model_name": "claude-3-7-sonnet-20250219"},
    "claude_haiku": {"type": "anthropic", "model_name": "claude-3-5-haiku-20241022"},

    "deepl": {"type": "deepl"},

    "gemini_flash": {"type": "gemini", "model_name": "models/gemini-2.0-flash"},
    "gemini_flash_lite": {"type": "gemini", "model_name": "models/gemini-2.0-flash-lite"},
}

###############################################################################
# 3) 현재 선택된 모델명 저장
###############################################################################
_current_model = None
_current_info = None

def set_model(model_name: str):
    info = MODEL_INFO.get(model_name)
    if not info:
        info = {"type": "openai", "model_name": "o3-mini"}
    global _current_model, _current_info
    _current_model = model_name
    _current_info = info
    return info

###############################################################################
# 전처리: HTML 태그, 이모지, 특수문자 제거
###############################################################################
def preprocess_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    return text.strip()

###############################################################################
# 4) 공용 함수: 청크 분할 (chunk size=200)
###############################################################################
def chunk_text(text, chunk_size=200):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def chunked_translate_o3(text, src_lang, tgt_lang):
    """
    긴 텍스트를 200자씩 잘라 번역.
    o3-mini 전용.
    """
    chunks = chunk_text(text, chunk_size=200)
    results = []
    for c in chunks:
        partial = _retry_openai_translate(c, src_lang, tgt_lang, "o3-mini")
        if not partial.strip():
            print(f"[Warning] Chunk 번역 실패: '{c}'")
            partial = "[번역 실패: 빈 응답]"
        results.append(partial)
    return " ".join(results)

###############################################################################
# _translate() : translation_eval.py에서 사용하는 엔트리
###############################################################################
def _translate(text, src_lang, tgt_lang, model_info=None):
    if model_info is None:
        print("[ERROR] model_info is None")
        return ""

    api_type = model_info["type"]
    model_id = model_info.get("model_name", "")

    if api_type == "openai":
        return _openai_translate(text, src_lang, tgt_lang, model_id)
    elif api_type == "anthropic":
        return _anthropic_translate(text, src_lang, tgt_lang, model_id)
    elif api_type == "deepl":
        return _deepl_translate(text, src_lang, tgt_lang)
    elif api_type == "gemini":
        return _gemini_translate(text, src_lang, tgt_lang, model_id)
    else:
        print(f"[ERROR] Unsupported api_type: {api_type}")
        return text

###############################################################################
# 5) OpenAI 번역
###############################################################################
def _retry_openai_translate(text, src_lang, tgt_lang, force_model_id="o3-mini", max_retry=5):
    """
    빈 응답 시 재시도 및 대체 프롬프트 시도
    """
    for attempt in range(max_retry):
        # 기본 프롬프트 시도
        resp = _openai_translate(text, src_lang, tgt_lang, force_model_id)
        if resp.strip():
            return resp
        print(f"[Retry {attempt+1}/{max_retry}] 빈 응답. 입력: '{text}'")
        time.sleep(2.0)
    return ""

def _openai_translate(text, src_lang, tgt_lang, model_id):
    if not OPENAI_API_KEY:
        print("[Warning] OPENAI_API_KEY not set.")
        return ""

    # o3-mini 전용 프롬프트: 독립적 처리 보장
    # prompt = (
    #     f"Ignore all previous conversations and context.\n"
    #     f"Translate the following text from {src_lang} to {tgt_lang} accurately and naturally.\n"
    #     f"Text: {text}\n"
    #     f"Return only the translated text without any explanation."
    # )
    prompt = (
        f"다음 문장을 '{src_lang}'에서 '{tgt_lang}'로 번역해 주세요. "
        "핵심 정보를 유지하고 자연스럽게 번역을 해주세요.\n\n"
        f"{text}"
    )
    # prompt = f"{text}"
    print("---> prompt :", prompt)

    try:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=1024
        )
        return response.choices[0].message.content.strip()

    except openai.BadRequestError as e:
        print(f"[OpenAI BadRequestError] {e} | 입력: '{text}' | 프롬프트: '{prompt}'")
        return ""
    except openai.RateLimitError as e:
        print(f"[OpenAI RateLimitError] {e} | 입력: '{text}' | 프롬프트: '{prompt}'")
        return ""
    except openai.APIError as e:
        print(f"[OpenAI APIError] {e} | 입력: '{text}' | 프롬프트: '{prompt}'")
        return ""
    except Exception as e:
        print(f"[ERROR] OpenAI API: {e} | 입력: '{text}' | 프롬프트: '{prompt}'")
        return ""

###############################################################################
# 6) Claude (Anthropic) 번역
###############################################################################
def _anthropic_translate(text, src_lang, tgt_lang, model_id):
    if not anthropic_client:
        print("[Warning] ANTHROPIC_API_KEY not set or anthropic_client is None.")
        return ""

    prompt = (
        f"다음 문장을 '{src_lang}'에서 '{tgt_lang}'로 번역해 주세요. "
        "핵심 정보를 유지하고 자연스럽게 번역을 해주세요.\n\n"
        f"{text}"
    )

    try:
        resp = anthropic_client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text.strip()
    except Exception as e:
        print(f"[ERROR] Claude API: {e}")
        return ""

###############################################################################
# 7) DeepL 번역
###############################################################################
def _deepl_translate(text, src_lang, tgt_lang):
    if not translator_deepl:
        print("[Warning] DEEPL_API_KEY not set or translator_deepl is None.")
        return ""

    text = preprocess_text(text)
    if tgt_lang.lower().startswith("en"):
        target_lang = "EN-US"
    elif tgt_lang.lower().startswith("ko"):
        target_lang = "KO"
    else:
        target_lang = "EN-US"

    try:
        result = translator_deepl.translate_text(text, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"[ERROR] DeepL API: {e}")
        return ""

###############################################################################
# 8) Gemini 번역
###############################################################################
def _gemini_translate(text, src_lang, tgt_lang, model_id):
    if not gemini_enabled or not GEMINI_API_KEY:
        print("[Warning] Gemini 사용 불가. API 키 누락 또는 모듈 미설치.")
        return ""

    text = preprocess_text(text)
    prompt = (
        f"다음 문장을 '{src_lang}'에서 '{tgt_lang}'로 번역해 주세요. "
        "핵심 정보를 유지하고 자연스럽게 번역해 주세요.\n\n"
        f"{text}"
    )
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini API: {e}")
        return ""