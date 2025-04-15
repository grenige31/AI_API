###############################################################################
# translation.py - BLEU + BERTScore 평가 포함 + ai_api.py 통합
###############################################################################
import json
import os
import re
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from kiwipiepy import Kiwi

# ✅ ai_api.py 연동
from ai_api import set_model, translate_ko2en, translate_en2ko, MODEL_INFO

# ✅ 입력/출력 디렉토리 경로 설정
input_file = os.path.join("data", "komt-1810k-test.jsonl")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

repeat_num = 2  # ✅ 모든 모델에 대해 동일하게 사용할 샘플 수

# ✅ punkt 오류 방지 (Colab용)
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.data.path.append("/root/nltk_data")

# ✅ 한국어 토크나이저
kiwi = Kiwi()

def is_korean(text):
    return any('가' <= char <= '힣' for char in text)

def tokenize(text):
    if is_korean(text):
        return [token.form for token in kiwi.tokenize(text, normalize_coda=True)]
    else:
        return text.lower().split()  # ✅ word_tokenize 대신 안전한 대체

def simple_score(reference_text, candidate_text):
    reference_tokens = tokenize(reference_text)
    candidate_tokens = tokenize(candidate_text)
    score_val = sentence_bleu([reference_tokens], candidate_tokens,
                               smoothing_function=SmoothingFunction().method2)
    return score_val

def load_json(filename):
    json_data = []
    with open(filename, "r", encoding="utf-8") as f:
        if os.path.splitext(filename)[1] != ".jsonl":
            json_data = json.load(f)
        else:
            for line in f:
                json_data.append(json.loads(line))
    return json_data

def save_json(json_data, filename, option="a"):
    filename = filename.replace(" ", "_")
    with open(filename, option, encoding="utf-8") as f:
        if filename.endswith(".jsonl"):
            for data in json_data:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

def run_translation(json_data, model_name, repeat_num=1, output_prefix="results", result_collector=None):
    set_model(model_name)

    for repeat in range(repeat_num):
        print(f"\n[Model: {model_name}] 반복 {repeat+1}/{repeat_num}")
        for index, data in tqdm(enumerate(json_data), total=len(json_data)):
            chat = data["conversations"]
            src = data["src"]
            input_text = chat[0]["value"]
            reference = chat[1]["value"]

            def clean_text(text):
                if "한글로 번역하세요." in text:
                    cur_lang = "en"
                else:
                    cur_lang = "ko"
                return text.split("번역하세요.\n", 1)[-1], cur_lang

            input_text, cur_lang = clean_text(input_text)

            if cur_lang == "en":
                generation = translate_en2ko(input_text)
            else:
                generation = translate_ko2en(input_text)

            # ✅ BLEU 계산
            bleu = round(simple_score(reference, generation), 3)

            # ✅ BERTScore 계산
            try:
                P, R, F1 = score([generation], [reference], lang="ko", model_type="bert-base-multilingual-cased")
                precision = round(P[0].item(), 4)
                recall = round(R[0].item(), 4)
                f1 = round(F1[0].item(), 4)
            except Exception as e:
                print(f"[ERROR] BERTScore 계산 실패: {e}")
                precision = recall = f1 = 0.0

            result = {
                "repeat": repeat + 1,
                "index": index,
                "reference": reference,
                "generation": generation,
                "bleu": bleu,
                "bertscore_precision": precision,
                "bertscore_recall": recall,
                "bertscore_f1": f1,
                "lang": cur_lang,
                "model": model_name,
                "src": src,
                "conversations": chat
            }

            print(json.dumps(result, ensure_ascii=False, indent=2))

            output_file = os.path.join(output_dir, f"{output_prefix}_{model_name}.jsonl")
            save_json([result], output_file)

            # ✅ 결과 수집
            if result_collector is not None:
                result_collector.append(result)

import matplotlib.pyplot as plt

def plot_model_scores(summary_list):
    models = [s["model"] for s in summary_list]
    bleu_scores = [s["average_bleu"] for s in summary_list]
    f1_scores = [s["average_bertscore_f1"] for s in summary_list]

    x = range(len(models))

    plt.figure(figsize=(12, 6))

    # BLEU
    plt.subplot(1, 2, 1)
    plt.bar(x, bleu_scores)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("Average BLEU Score by Model")
    plt.ylabel("BLEU Score")

    # BERTScore F1
    plt.subplot(1, 2, 2)
    plt.bar(x, f1_scores, color='orange')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("Average BERTScore F1 by Model")
    plt.ylabel("BERTScore F1")

    plt.tight_layout()
    plt.show()

def main():
    json_data = load_json(input_file)
    all_model_results = {}

    for model_name in MODEL_INFO:
        all_results = []
        run_translation(json_data, model_name, repeat_num=repeat_num, result_collector=all_results)
        all_model_results[model_name] = all_results

    print("\n\n📊 === 모델별 평균 평가 결과 ===")
    summaries = []  # ✅ 저장용

    for model_name, results in all_model_results.items():
        if not results:
            continue
        avg_bleu = sum(r["bleu"] for r in results) / len(results)
        avg_precision = sum(r["bertscore_precision"] for r in results) / len(results)
        avg_recall = sum(r["bertscore_recall"] for r in results) / len(results)
        avg_f1 = sum(r["bertscore_f1"] for r in results) / len(results)

        summary = {
            "model": model_name,
            "average_bleu": round(avg_bleu, 4),
            "average_bertscore_precision": round(avg_precision, 4),
            "average_bertscore_recall": round(avg_recall, 4),
            "average_bertscore_f1": round(avg_f1, 4)
        }

        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ✅ results.jsonl로 저장
    save_json(summaries, os.path.join(output_dir, "results.jsonl"), option="w")

    # ✅ 시각화
    plot_model_scores(summaries)


if __name__ == "__main__":
    main()
