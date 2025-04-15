###############################################################################
#ai_api.py
#  - 하나의 파일 안에서 openai, claude, deepL, gemini 등 다양한 API를 선택적으로 사용
###############################################################################
import os
import openai
import deepl
from anthropic import Client

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
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

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
    global _current_model, _current_info
    _current_model = model_name
    _current_info = MODEL_INFO.get(model_name)
    if not _current_info:
        print(f"[Warning] 모델명 '{model_name}'은 MODEL_INFO에 정의되지 않았습니다. fallback=OpenAI gpt-3.5")
        _current_info = {"type": "openai", "model_name": "gpt-3.5-turbo"}

###############################################################################
# 4) 공용 함수
###############################################################################
def translate_ko2en(text: str) -> str:
    if not _current_info:
        set_model("openai_gpt_4o")
    return _translate(text, src_lang="ko", tgt_lang="en")

def translate_en2ko(text: str) -> str:
    if not _current_info:
        set_model("openai_gpt_4o")
    return _translate(text, src_lang="en", tgt_lang="ko")

def _translate(text, src_lang, tgt_lang):
    api_type = _current_info["type"]
    model_id = _current_info.get("model_name", "")

    if api_type == "openai":
        return _openai_translate(text, src_lang, tgt_lang, model_id)
    elif api_type == "anthropic":
        return _anthropic_translate(text, src_lang, tgt_lang, model_id)
    elif api_type == "deepl":
        return _deepl_translate(text, src_lang, tgt_lang)
    elif api_type == "gemini":
        return _gemini_translate(text, src_lang, tgt_lang, model_id)
    else:
        return text

###############################################################################
# 5) OpenAI 번역
###############################################################################
def _openai_translate(text, src_lang, tgt_lang, model_id):
    if not OPENAI_API_KEY:
        print("[Warning] OPENAI_API_KEY not set.")
        return ""

    prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{text}"
    try:

        # 1
        # response = openai.ChatCompletion.create(
        #     model=model_id,
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=512,
        #     temperature=0.7,
        # )

        #2
        # kwargs = {
        #     "model": model_id,
        #     "messages": [{"role": "user", "content": prompt}],
        # }
        #_________________________________________________

        # # 최신 OpenAI 모델: 'temperature', 'max_tokens' 안됨
        # if model_id.startswith("gpt-4o") or "mini" in model_id or "o3" in model_id:
        #     kwargs["max_completion_tokens"] = 512
        # else:
        #     kwargs["temperature"] = 0.7
        #     kwargs["max_tokens"] = 512

        # response = openai.ChatCompletion.create(**kwargs)
        #_________________________________________________

        #3
        kwargs = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}]
        }

        # o1, o3 계열 모델은 새로운 파라미터만 지원
        if "o1" in model_id or "o3" in model_id:
            kwargs["max_completion_tokens"] = 512
        else:
            kwargs["max_tokens"] = 512
            kwargs["temperature"] = 0.7

        response = openai.chat.completions.create(**kwargs)
        #_________________________________________________

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] OpenAI API: {e}")
        return ""

###############################################################################
# 6) Claude (Anthropic) 번역
###############################################################################
def _anthropic_translate(text, src_lang, tgt_lang, model_id):
    if not anthropic_client:
        print("[Warning] ANTHROPIC_API_KEY not set or anthropic_client is None.")
        return ""

    prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{text}"
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

    if tgt_lang.lower().startswith("en"):
        target_lang = "EN-US"  # ✅ 변경됨
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

    prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{text}"
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[ERROR] Gemini API: {e}")
        return ""
