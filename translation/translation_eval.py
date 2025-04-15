import json
import os
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
from kiwipiepy import Kiwi
import time
import matplotlib.pyplot as plt

from ai_api import set_model, MODEL_INFO, _translate

input_file = os.path.join("data", "komt-1810k-test.jsonl")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

repeat_num = 3

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.data.path.append("/root/nltk_data")

kiwi = Kiwi()

def is_korean(text):
    return any('가' <= char <= '힣' for char in text)

def tokenize(text):
    if is_korean(text):
        return [token.form for token in kiwi.tokenize(text, normalize_coda=True)]
    else:
        return text.lower().split()

def simple_score(reference_text, candidate_text):
    ref_tokens = tokenize(reference_text)
    gen_tokens = tokenize(candidate_text)
    return sentence_bleu(
        [ref_tokens],
        gen_tokens,
        smoothing_function=SmoothingFunction().method2
    )

def load_json(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        if os.path.splitext(filename)[1] != ".jsonl":
            data = json.load(f)
        else:
            for line in f:
                data.append(json.loads(line))
    return data

def save_json(json_data, filename, option="a"):
    filename = filename.replace(" ", "_")
    with open(filename, option, encoding="utf-8") as f:
        if filename.endswith(".jsonl"):
            for d in json_data:
                json.dump(d, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

def run_translation(json_data, model_name, repeat_num=1, result_collector=None):
    info = set_model(model_name)
    for rpt in range(repeat_num):
        print(f"\n[Model: {model_name}] 반복 {rpt+1}/{repeat_num}")
        for idx, record in tqdm(enumerate(json_data), total=len(json_data)):
            chat = record.get("conversations", [])
            src = record.get("src", "")

            input_text = chat[0].get("value", "")
            reference = chat[1].get("value", "")

            # def clean_text(text):
            #     lower = text.lower()
            #     cleaned_text = text
            #     if "영어로 번역" in lower:
            #         cur_lang = "ko"
            #         cleaned_text = text.split("영어로 번역하세요.\n", 1)[-1].strip()
            #     elif "한글로 번역" in lower or "한국어로 번역" in lower:
            #         cur_lang = "en"
            #         cleaned_text = text.split("번역하세요.\n", 1)[-1].strip()
            #     else:
            #         cleaned_text = text.strip()
            #         cur_lang = "ko" if is_korean(cleaned_text) else "en"
            #     print(f"[Debug] 원본 텍스트: '{text}', 처리된 텍스트: '{cleaned_text}', 감지된 언어: {cur_lang}")
            #     return cleaned_text.strip(), cur_lang
            def clean_text(text):
                if chat[0]["value"].find("한글로 번역하세요.") != -1:
                    cur_lang = "en"
                
                else:
                    cur_lang = "ko"
                text = text.split("번역하세요.\n", 1)[-1]
                return text, cur_lang

            input_text, cur_lang = clean_text(input_text)
            tar_lang="ko"
            if cur_lang=="ko":
                tar_lang="en"
            if not input_text:
                print(f"[Warning] 빈 입력 텍스트: index={idx}")
                generation = "[번역 실패: 빈 입력]"
            else:
                generation = ""
                for attempt in range(3):

                    generation = _translate(input_text, cur_lang, tar_lang, model_info=info)
                    if generation.strip() and generation != "[번역 실패: 빈 응답]":
                        break
                    print(f"[Retry {attempt+1}/3] 빈 응답 => 재시도 | 입력: '{input_text}'")
                    time.sleep(1.0)

                if not generation.strip():
                    generation = "[번역 실패: 빈 응답]"

            bleu_val = round(simple_score(reference, generation), 3)

            try:
                P, R, F1 = score([generation], [reference], lang="ko", model_type="bert-base-multilingual-cased")
                prec = round(P[0].item(), 4)
                rec = round(R[0].item(), 4)
                f1_val = round(F1[0].item(), 4)
            except Exception as e:
                print(f"[ERROR] BERTScore 실패: {e} | 입력: '{input_text}', 생성: '{generation}'")
                prec = rec = f1_val = 0.0

            result = {
                "repeat": rpt+1,
                "index": idx,
                "reference": reference,
                "generation": generation,
                "bleu": bleu_val,
                "bertscore_precision": prec,
                "bertscore_recall": rec,
                "bertscore_f1": f1_val,
                "lang": cur_lang,
                "model": model_name,
                "src": src,
                "conversations": chat
            }

            print(json.dumps(result, ensure_ascii=False, indent=2))

            out_file = os.path.join(output_dir, f"results_{model_name}.jsonl")
            save_json([result], out_file)

            if result_collector is not None:
                result_collector.append(result)

def plot_model_scores(summary_list):
    models = [s["model"] for s in summary_list]
    bleu_scores = [s["average_bleu"] for s in summary_list]
    f1_scores = [s["average_bertscore_f1"] for s in summary_list]

    x = range(len(models))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(x, bleu_scores)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("Average BLEU Score by Model")
    plt.ylabel("BLEU Score")

    plt.subplot(1, 2, 2)
    plt.bar(x, f1_scores, color='orange')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("Average BERTScore F1 by Model")
    plt.ylabel("BERTScore F1")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_scores.png"))
    plt.close()

def main():
    data = load_json(input_file)
    all_model_results = {}

    for model_nm in MODEL_INFO:
        results_for_this_model = []
        run_translation(data, model_nm, repeat_num=repeat_num, result_collector=results_for_this_model)
        all_model_results[model_nm] = results_for_this_model

    print("\n\n📊 === 모델별 평균 평가 결과 ===")
    summaries = []
    for model_nm, results in all_model_results.items():
        if not results:
            continue
        avg_bleu = sum(r["bleu"] for r in results)/len(results)
        avg_prec = sum(r["bertscore_precision"] for r in results)/len(results)
        avg_rec = sum(r["bertscore_recall"] for r in results)/len(results)
        avg_f1 = sum(r["bertscore_f1"] for r in results)/len(results)

        summary = {
            "model": model_nm,
            "average_bleu": round(avg_bleu, 4),
            "average_bertscore_precision": round(avg_prec, 4),
            "average_bertscore_recall": round(avg_rec, 4),
            "average_bertscore_f1": round(avg_f1, 4)
        }
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    save_json(summaries, os.path.join(output_dir, "results_all_model.jsonl"), option="w")
    plot_model_scores(summaries)

if __name__ == "__main__":
    main()
