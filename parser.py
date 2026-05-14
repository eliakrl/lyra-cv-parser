import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from markitdown import MarkItDown
from transformers import pipeline

load_dotenv()

_ner = None

def _get_ner():
    global _ner
    if _ner is None:
        _ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return _ner


def extract_text(file_path: str) -> str:
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        md = MarkItDown()
        result = md.convert(file_path)
        text = result.text_content.strip()
    print(f"[debug] extracted {len(text)} chars: {repr(text[:200])}")
    if len(text) < 20:
        raise ValueError("Extracted text too short — file may be a scanned image. Try OCR.")
    return text


def extract_entities(text: str) -> dict:
    ner = _get_ner()
    # DistilBERT has a 512-token limit; chunk if needed
    chunks = [text[i:i+1000] for i in range(0, min(len(text), 4000), 1000)]
    orgs, persons = set(), set()
    for chunk in chunks:
        for ent in ner(chunk):
            if ent["entity_group"] == "ORG":
                orgs.add(ent["word"].strip())
            elif ent["entity_group"] == "PER":
                persons.add(ent["word"].strip())
    return {"persons": list(persons), "organizations": list(orgs)}


def extract_years_experience(text: str) -> int | None:
    patterns = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?experience",
        r"experience\s*(?:of\s*)?(\d+)\+?\s*years?",
        r"(\d+)\+?\s*años?\s*de\s*experiencia",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def extract_date_spans(text: str) -> list[str]:
    # Matches patterns like "Jan 2020 – Mar 2022" or "2018 - 2021"
    pattern = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)?\s*\d{4}\s*[-–—]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|present|presente|hoy|today)?\s*\d{0,4}"
    return re.findall(pattern, text, re.IGNORECASE)


def run_llama(text: str) -> dict:
    prompt = f"""You are a CV parser. Extract the following from this CV text and return ONLY valid JSON, no explanation:
{{
  "seniority": "junior | mid | senior | lead",
  "transferable_skills": ["list of soft skills mentioned or implied"],
  "career_gaps": ["list any employment gaps longer than 6 months, e.g. '2019-2020'"],
  "languages": ["languages the candidate speaks"]
}}

CV:
{text[:3000]}"""

    try:
        client = OpenAI(
            api_key=os.getenv("NEBIUS_API_KEY"),
            base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1"),
        )
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=[
                {"role": "system", "content": "You are a structured CV parser. Your only job is to extract factual information from CV text and return it as valid JSON. Never explain, never add commentary. If a field cannot be determined from the text, return null for strings or [] for arrays. Be conservative — only infer what the text clearly supports."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except Exception as e:
        return {"error": f"Nebius unavailable: {e}"}


def parse_cv(file_path: str, use_llama: bool = False) -> dict:
    text = extract_text(file_path)
    entities = extract_entities(text)
    result = {
        "file": file_path,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "organizations": entities["organizations"],
        "persons": entities["persons"],
        "years_experience": extract_years_experience(text),
        "date_spans": extract_date_spans(text),
    }
    if use_llama:
        result["soft_inference"] = run_llama(text)
    return result
