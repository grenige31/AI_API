
###############################################################################
# ai_api.py
#  - OpenAI, Anthropic, Gemini 등 다양한 API를 선택적으로 사용하며
#    요약 기능을 제공합니다.
###############################################################################
import os
import openai
from anthropic import Client

# Gemini API 초기화
try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        print("DEBUG: Configuring Gemini with provided API key.")
        genai.configure(api_key=GEMINI_API_KEY)
    gemini_enabled = True
except ImportError:
    gemini_enabled = False
    print("[Warning] google-generativeai 패키지가 설치되지 않았습니다. Gemini 사용 불가.")

###############################################################################
# 1) 환경 변수로 API 키 읽기
###############################################################################
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

print("DEBUG: API KEYs loaded:")
print(f"  OPENAI_API_KEY: {'set' if OPENAI_API_KEY else 'not set'}")
print(f"  ANTHROPIC_API_KEY: {'set' if ANTHROPIC_API_KEY else 'not set'}")

# API 키 설정
openai.api_key = OPENAI_API_KEY
anthropic_client = Client(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

###############################################################################
# 2) 모델 정보 사전
###############################################################################
MODEL_INFO = {
    # "openai_gpt_4o": {"type": "openai", "model_name": "gpt-4o"},
    # "openai_gpt_4o_mini": {"type": "openai", "model_name": "gpt-4o-mini"},
    "openai_o3_mini": {"type": "openai", "model_name": "o3-mini"},
    # "claude_sonnet": {"type": "anthropic", "model_name": "claude-3-7-sonnet-20250219"},
    # "claude_haiku": {"type": "anthropic", "model_name": "claude-3-5-haiku-20241022"},
    # "gemini_flash": {"type": "gemini", "model_name": "models/gemini-2.0-flash"},
    # "gemini_flash_lite": {"type": "gemini", "model_name": "models/gemini-2.0-flash-lite"},
}

###############################################################################
# 3) 현재 선택된 모델 상태 저장
###############################################################################
_current_model = None
_current_info = None

def set_model(model_name: str):
    """사용할 모델을 설정하고 내부 상태를 업데이트합니다."""
    global _current_model, _current_info
    print(f"DEBUG: Setting model to '{model_name}'")
    _current_model = model_name
    _current_info = MODEL_INFO.get(model_name)
    if not _current_info:
        print(f"[Warning] 모델명 '{model_name}'은 MODEL_INFO에 정의되지 않았습니다. fallback=OpenAI gpt-3.5")
        _current_info = {"type": "openai", "model_name": "gpt-3.5-turbo"}
    print(f"DEBUG: Model set. Type: {_current_info['type']}, Model Name: {_current_info.get('model_name', '')}")

###############################################################################
# 4) 요약 기능
###############################################################################
def summarize(text: str) -> str:
    """
    입력 텍스트를 요약하는 함수.
    선택한 API의 요약 기능을 호출합니다.
    """
    if not _current_info:
        set_model("openai_gpt_4o")
    api_type = _current_info["type"]
    model_id = _current_info.get("model_name", "")
    print(f"DEBUG: Summarize using API type: {api_type}, model_id: {model_id}")

    if api_type == "openai":
        return _openai_summarize(text, model_id)
    elif api_type == "anthropic":
        return _anthropic_summarize(text, model_id)
    elif api_type == "gemini":
        return _gemini_summarize(text, model_id)
    else:
        print("DEBUG: Unknown API type in summarize. Returning original text.")
        return text
def _openai_summarize(text, model_id):
    if not OPENAI_API_KEY:
        print("[Warning] OPENAI_API_KEY not set.")
        return ""

    prompt = f"Summarize the following text:\n{text}"

    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
    }
    if "o1" in model_id or "o3" in model_id:
      print("DEBUG: Detected o1/o3 model")
      kwargs["max_completion_tokens"] = 512  # ✅ 이걸로만 설정
      # 'response_format' 제거
    else:
      print("DEBUG: Detected 4o or other model")
      kwargs["max_tokens"] = 512
      kwargs["temperature"] = 0.7

    try:
        response = openai.chat.completions.create(**kwargs)
        result = response.choices[0].message.content.strip()
        if not result:
            print("[Warning] Empty summary generated.")
        return result
    except Exception as e:
        print(f"[ERROR] OpenAI API Summarize: {e}")
        return ""

def _anthropic_summarize(text, model_id):
    print("DEBUG: _anthropic_summarize called.")
    if not anthropic_client:
        print("[Warning] ANTHROPIC_API_KEY not set or anthropic_client is None.")
        return ""
    prompt = f"Summarize the following text:\n{text}"
    try:
        resp = anthropic_client.messages.create(
            model=model_id,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        result = resp.content[0].text.strip()
        print("DEBUG: _anthropic_summarize successful.")
        return result
    except Exception as e:
        print(f"[ERROR] Claude API Summarize: {e}")
        return ""

def _gemini_summarize(text, model_id):
    print("DEBUG: _gemini_summarize called.")
    if not gemini_enabled or not GEMINI_API_KEY:
        print("[Warning] Gemini 사용 불가. API 키 누락 또는 모듈 미설치.")
        return ""
    prompt = f"Summarize the following text:\n{text}"
    try:
        model = genai.GenerativeModel(model_id)
        response = model.generate_content(prompt)
        result = response.text.strip()
        print("DEBUG: _gemini_summarize successful.")
        return result
    except Exception as e:
        print(f"[ERROR] Gemini API Summarize: {e}")
        return ""



