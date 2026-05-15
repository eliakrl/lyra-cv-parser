import re
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from markitdown import MarkItDown
from transformers import pipeline
import pytesseract
from pdf2image import convert_from_path

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


def _ocr_pdf(file_path: str) -> str:
    pages = convert_from_path(file_path, dpi=300)
    return "\n".join(pytesseract.image_to_string(page) for page in pages).strip()


def extract_text(file_path: str) -> str:
    if file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    else:
        md = MarkItDown()
        result = md.convert(file_path)
        text = result.text_content.strip()

    print(f"[debug] extracted {len(text)} chars via MarkItDown: {repr(text[:200])}")

    if len(text) < 20 and file_path.lower().endswith('.pdf'):
        print("[debug] falling back to Tesseract OCR")
        text = _ocr_pdf(file_path)
        print(f"[debug] OCR yielded {len(text)} chars: {repr(text[:200])}")

    if len(text) < 20:
        raise ValueError("Extracted text too short — could not read file content.")

    return text


def _is_clean_entity(word: str, score: float, min_score: float = 0.85) -> bool:
    if score < min_score:
        return False
    if "##" in word:
        return False
    if len(word) < 4:
        return False
    if word == word.lower():
        return False
    return True


def extract_entities(text: str) -> dict:
    ner = _get_ner()
    chunks = [text[i:i+1000] for i in range(0, min(len(text), 4000), 1000)]
    orgs_confident, orgs_uncertain, persons = set(), set(), set()
    for chunk in chunks:
        for ent in ner(chunk):
            word = ent["word"].strip()
            score = ent["score"]
            if ent["entity_group"] == "ORG":
                if _is_clean_entity(word, score, min_score=0.85):
                    orgs_confident.add(word)
                elif _is_clean_entity(word, score, min_score=0.6):
                    orgs_uncertain.add(word)
            elif ent["entity_group"] == "PER" and _is_clean_entity(word, score):
                persons.add(word)
    return {
        "persons": list(persons),
        "organizations": list(orgs_confident),
        "organizations_uncertain": list(orgs_uncertain - orgs_confident),
    }


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


_MONTH_MAP = {
    "jan": "01", "january": "01", "enero": "01", "gennaio": "01", "januar": "01",
    "feb": "02", "february": "02", "febrero": "02", "febbraio": "02", "februar": "02",
    "mar": "03", "march": "03", "marzo": "03", "märz": "03",
    "apr": "04", "april": "04", "abril": "04", "aprile": "04",
    "may": "05", "mayo": "05", "maggio": "05", "mai": "05",
    "jun": "06", "june": "06", "junio": "06", "giugno": "06", "juni": "06",
    "jul": "07", "july": "07", "julio": "07", "luglio": "07", "juli": "07",
    "aug": "08", "august": "08", "agosto": "08",
    "sep": "09", "sept": "09", "september": "09", "septiembre": "09", "settembre": "09",
    "oct": "10", "october": "10", "octubre": "10", "ottobre": "10", "oktober": "10",
    "nov": "11", "november": "11", "noviembre": "11", "novembre": "11",
    "dec": "12", "december": "12", "diciembre": "12", "dicembre": "12", "dezember": "12",
}
_PRESENT_WORDS = {"present", "presente", "hoy", "today", "heute", "ora", "now", "actuel", "aujourd'hui"}


def _parse_date_part(part: str) -> str | None:
    part = part.strip()
    if part.lower() in _PRESENT_WORDS or part == "":
        return "present"
    year_match = re.search(r"\d{4}", part)
    if not year_match:
        return None
    year = year_match.group()
    month = "01"
    word_match = re.search(r"[a-zA-ZäöüÄÖÜß]+", part)
    if word_match:
        month = _MONTH_MAP.get(word_match.group().lower(), "01")
    return f"{month}-{year}"


def _normalize_span(raw: str) -> dict:
    raw = raw.strip()
    parts = re.split(r"\s*[-–—/]\s*|\s+to\s+|\s+bis\s+|\s+a\s+", raw, maxsplit=1)
    if len(parts) == 2:
        frm = _parse_date_part(parts[0])
        to = _parse_date_part(parts[1])
    else:
        frm = _parse_date_part(parts[0])
        to = None
    return {"from": frm, "to": to, "raw": raw}


def extract_date_spans(text: str) -> list[dict]:
    pattern = r"(?:Jan(?:uary|vier)?|Feb(?:ruary|ruar)?|M(?:ar(?:ch|zo|z)?|ai|ay|aggio)|Apr(?:il|e)?|Jun(?:e|io|i)?|Jul(?:y|io|i)?|Aug(?:ust|osto)?|Sep(?:t(?:ember|iembre|embre)?)?|O(?:ct(?:ober|ubre|tobre)?|ktober)|Nov(?:ember|iembre|embre)?|D(?:ec(?:ember|iembre|embre)?|ez(?:ember)?))\s+\d{4}\s*[-–—]\s*(?:Jan(?:uary|vier)?|Feb(?:ruary|ruar)?|M(?:ar(?:ch|zo|z)?|ai|ay|aggio)|Apr(?:il|e)?|Jun(?:e|io|i)?|Jul(?:y|io|i)?|Aug(?:ust|osto)?|Sep(?:t(?:ember|iembre|embre)?)?|O(?:ct(?:ober|ubre|tobre)?|ktober)|Nov(?:ember|iembre|embre)?|D(?:ec(?:ember|iembre|embre)?|ez(?:ember)?)|present|presente|hoy|today|heute|now)(?:\s+\d{4})?|\d{4}\s*[-–—]\s*(?:\d{4}|present|presente|hoy|today|heute|now)|seit\s+\d{4}"
    raw_matches = re.findall(pattern, text, re.IGNORECASE)
    return [_normalize_span(r) for r in raw_matches]


def _call_nebius(prompt: str) -> dict:
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
            timeout=30,
        )
        raw = response.choices[0].message.content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except Exception as e:
        return {"error": f"Nebius unavailable: {e}"}


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
    return _call_nebius(prompt)


def run_llama_full(text: str) -> dict:
    clean = " ".join(text.split())
    prompt = f"""You are a CV parser. Extract ALL of the following from this CV text and return ONLY valid JSON, no explanation:
{{
  "seniority": "junior | mid | senior | lead",
  "organizations": ["only employer and company names the candidate worked at — exclude schools, tools, and skills"],
  "years_experience": 0,
  "career_gaps": ["gaps between jobs longer than 6 months as 'YYYY-MM to YYYY-MM'"],
  "languages": ["languages the candidate speaks"],
  "transferable_skills": ["soft skills mentioned or implied"]
}}

For years_experience: calculate from date spans in the CV if not stated explicitly. Return an integer or null.
For organizations: be strict — only actual employers, not software tools or institutions.

CV:
{clean[:2500]}"""
    return _call_nebius(prompt)


def parse_cv(file_path: str, mode: str = "hybrid") -> dict:
    text = extract_text(file_path)

    if mode == "llama":
        llama = run_llama_full(text)
        return {
            "file": file_path,
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "organizations": llama.get("organizations", []),
            "persons": [],
            "years_experience": llama.get("years_experience"),
            "date_spans": extract_date_spans(text),
            "mode": "llama",
            "soft_inference": {
                "seniority": llama.get("seniority"),
                "transferable_skills": llama.get("transferable_skills", []),
                "career_gaps": llama.get("career_gaps", []),
                "languages": llama.get("languages", []),
            },
        }

    entities = extract_entities(text)
    result = {
        "file": file_path,
        "text_preview": text[:300] + "..." if len(text) > 300 else text,
        "organizations": entities["organizations"],
        "organizations_uncertain": entities["organizations_uncertain"],
        "persons": entities["persons"],
        "years_experience": extract_years_experience(text),
        "date_spans": extract_date_spans(text),
        "mode": "hybrid",
        "soft_inference": run_llama(text),
    }
    return result
