import json
import os
from collections import OrderedDict
from tqdm import tqdm
from bert_score import score
from rouge import Rouge
import matplotlib.pyplot as plt
# ✅ ai_api.py 연동
from ai_api import set_model, MODEL_INFO, summarize


repeat_num = 2

input_file = os.path.join("data", "news200.jsonl")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

rouge = Rouge()

def load_json(filename):
    print(f"DEBUG: Loading data from {filename}")
    json_data = []
    with open(filename, "r", encoding="utf-8") as f:
        if os.path.splitext(filename)[1] != ".jsonl":
            json_data = json.load(f)
        else:
            for line in f:
                json_data.append(json.loads(line))
    print(f"DEBUG: Loaded {len(json_data)} records")
    return json_data

def save_json(json_data, filename, option="a"):
    directory, _ = os.path.split(filename)
    if directory and not os.path.exists(directory):
        print(f"DEBUG: Creating directory {directory}")
        os.makedirs(directory)
    filename = filename.replace(" ", "_")
    with open(filename, option, encoding="utf-8") as f:
        if filename.endswith(".jsonl"):
            for data in json_data:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        else:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"DEBUG: Saved {len(json_data)} record(s) to {filename}")

def plot_bert_score_summary(summary_list):
    models = [item["model"] for item in summary_list]
    precisions = [item["average_bertscore"]["precision"] for item in summary_list]
    recalls = [item["average_bertscore"]["recall"] for item in summary_list]
    f1s = [item["average_bertscore"]["f1"] for item in summary_list]

    x = range(len(models))

    plt.figure(figsize=(12, 6))

    # Precision
    plt.subplot(1, 3, 1)
    plt.bar(x, precisions)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("BERTScore Precision")
    plt.ylim(0, 1)

    # Recall
    plt.subplot(1, 3, 2)
    plt.bar(x, recalls, color='orange')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("BERTScore Recall")
    plt.ylim(0, 1)

    # F1
    plt.subplot(1, 3, 3)
    plt.bar(x, f1s, color='green')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("BERTScore F1")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

def plot_rouge_score_summary(summary_list):
    models = [item["model"] for item in summary_list]
    rouge1s = [item["average_rouge"]["rouge-1"] for item in summary_list]
    rouge2s = [item["average_rouge"]["rouge-2"] for item in summary_list]
    rougels = [item["average_rouge"]["rouge-l"] for item in summary_list]

    x = range(len(models))

    plt.figure(figsize=(12, 6))

    # ROUGE-1
    plt.subplot(1, 3, 1)
    plt.bar(x, rouge1s)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("ROUGE-1 F1")
    plt.ylim(0, 1)

    # ROUGE-2
    plt.subplot(1, 3, 2)
    plt.bar(x, rouge2s, color='orange')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("ROUGE-2 F1")
    plt.ylim(0, 1)

    # ROUGE-L
    plt.subplot(1, 3, 3)
    plt.bar(x, rougels, color='green')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.title("ROUGE-L F1")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()

def main():
    json_data = load_json(input_file)

    if not json_data:
        print("ERROR: 입력 데이터가 없습니다.")
        return

    all_model_avg_results = []

    for model_name in MODEL_INFO.keys():
        print(f"\n=== [Model: {model_name}] 전체 기사에 대해 {repeat_num}회 테스트 시작 ===")
        output_file = f"results_bert_{model_name}.jsonl"
        set_model(model_name)

        candidates = []
        references = []
        rouge1_all, rouge2_all, rougel_all = [], [], []

        for data_index, data in enumerate(json_data):
            article_list = data.get("article_original", [])
            article_text = " ".join(article_list) if isinstance(article_list, list) else ""
            abstractive = data.get("abstractive", "")

            if not article_text or not abstractive:
                print(f"DEBUG: Record {data_index} has missing content. Skipping.")
                continue

            for i in range(repeat_num):
                print(f"  → [Doc {data_index+1}/{len(json_data)}] 반복 {i+1}/{repeat_num}")
                generated_summary = summarize(article_text)
                print(f"    요약 결과: {generated_summary[:100]}...")

                candidates.append(generated_summary)
                references.append(abstractive)

                P, R, F1 = score([generated_summary], [abstractive], lang="ko", model_type="bert-base-multilingual-cased")
                rouge_scores = rouge.get_scores(generated_summary, abstractive)[0]

                rouge1_all.append(rouge_scores["rouge-1"]["f"])
                rouge2_all.append(rouge_scores["rouge-2"]["f"])
                rougel_all.append(rouge_scores["rouge-l"]["f"])

                result = OrderedDict()
                result["id"] = data.get("id", "")
                result["article_original"] = article_text
                result["abstractive"] = abstractive
                result["generated_summary"] = generated_summary
                result["model"] = model_name
                result["bertscore_precision"] = round(P[0].item(), 4)
                result["bertscore_recall"] = round(R[0].item(), 4)
                result["bertscore_f1"] = round(F1[0].item(), 4)
                result["rouge_1"] = rouge_scores["rouge-1"]["f"]
                result["rouge_2"] = rouge_scores["rouge-2"]["f"]
                result["rouge_l"] = rouge_scores["rouge-l"]["f"]
                output_prefix="results"
                print(json.dumps(result, ensure_ascii=False, indent=2))
                output_file = os.path.join(output_dir, f"{output_prefix}_{model_name}.jsonl")

                save_json([result], output_file)

        # 모델별 평균 계산
        if candidates and references:
            P_avg, R_avg, F1_avg = score(candidates, references, lang="ko", model_type="bert-base-multilingual-cased")
            avg_result = OrderedDict()
            avg_result["model"] = model_name
            avg_result["average_bertscore"] = {
                "precision": round(P_avg.mean().item(), 4),
                "recall": round(R_avg.mean().item(), 4),
                "f1": round(F1_avg.mean().item(), 4)
            }
            avg_result["average_rouge"] = {
                "rouge-1": round(sum(rouge1_all)/len(rouge1_all), 4),
                "rouge-2": round(sum(rouge2_all)/len(rouge2_all), 4),
                "rouge-l": round(sum(rougel_all)/len(rougel_all), 4)
            }

            print("\n=== [Model: {}] 전체 평균 BERTScore & ROUGE ===".format(model_name))
            print(json.dumps(avg_result, ensure_ascii=False, indent=2))
            save_json([avg_result], output_file)
            all_model_avg_results.append(avg_result)
        else:
            print(f"WARNING: 모델 {model_name}은 유효한 데이터가 없어 평균을 계산하지 않습니다.")

    # 전체 모델 평균 출력
    print("\n\n======= 전체 모델 평균 평가 요약 =======")
    for avg_result in all_model_avg_results:
        print(f"\n=== [Model: {avg_result['model']}] 평균 점수 ===")
        print(json.dumps(avg_result, ensure_ascii=False, indent=2))

    save_json(all_model_avg_results, "results_bert_all_models_summary.json", option="w")
    # ✅ 시각화 호출
    plot_bert_score_summary(all_model_avg_results)
    plot_rouge_score_summary(all_model_avg_results)

if __name__ == "__main__":
    main()