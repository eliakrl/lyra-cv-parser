import argparse
import json
from parser import parse_cv

def main():
    arg_parser = argparse.ArgumentParser(description="Lyra CV parser — PDF/DOCX to structured JSON")
    arg_parser.add_argument("cv_path", help="Path to CV file (PDF or DOCX)")
    arg_parser.add_argument("--llama", action="store_true", help="Use local Llama (Ollama) for soft inference: seniority, gaps, languages")
    args = arg_parser.parse_args()

    result = parse_cv(args.cv_path, use_llama=args.llama)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
