import pandas as pd
import json
import argparse
import os


def csv_to_json(csv_path, json_path=None):
    """
    CSV 파일을 LoRA 학습용 JSON 포맷으로 변환합니다.
    """
    df = pd.read_csv(csv_path)  # columns: filename, caption
    captions_dict = dict(zip(df["filename"], df["caption"]))

    if json_path and json_path.lower() != "none":
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(captions_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON: {json_path}")
    else:
        print("⚠️ JSON 저장 경로가 지정되지 않아 저장하지 않습니다.")

    return captions_dict


def json_to_jsonl(json_input, jsonl_path=None):
    """
    {file_name: text} 형식의 JSON을 Hugging Face 호환 JSONL로 변환합니다.
    """
    if isinstance(json_input, str):
        with open(json_input, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = json_input

    if jsonl_path and jsonl_path.lower() != "none":
        with open(jsonl_path, "w", encoding="utf-8") as f_out:
            if isinstance(metadata, dict):
                for fname, text in metadata.items():
                    json.dump({"file_name": fname, "text": text}, f_out)
                    f_out.write("\n")
            elif isinstance(metadata, list):
                for item in metadata:
                    json.dump(item, f_out)
                    f_out.write("\n")
            else:
                raise ValueError("❌ 지원하지 않는 JSON 포맷입니다.")
        print(f"✅ Saved JSONL: {jsonl_path}")
    else:
        print("⚠️ JSONL 저장 경로가 지정되지 않아 저장하지 않습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="입력 CSV 경로")
    parser.add_argument("--json", type=str, default=None, help="출력 JSON 경로 (LoRA 학습용)")
    parser.add_argument("--jsonl", type=str, default=None, help="출력 JSONL 경로 (Hugging Face Datasets용)")
    args = parser.parse_args()

    # CSV → JSON
    captions = csv_to_json(args.csv, args.json)

    # JSON → JSONL
    json_to_jsonl(captions, args.jsonl)

    # 임시 파일 삭제
    if args.json and args.json.lower() == "none" and os.path.exists("captions.json"):
        os.remove("captions.json")
