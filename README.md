# Lyra CV Parser

A prototype CV intelligence pipeline built for [Lyra](https://www.justlyra.com/#how), a platform that connects women with job opportunities. The pipeline extracts structured candidate profiles from uploaded CVs and complements them with a personality assessment questionnaire.

---

## Features

- **Text extraction** — MarkItDown converts digital-native PDFs and DOCX files to plain text locally, with no data transmitted
- **OCR fallback** — scanned or image-based PDFs are processed with Tesseract via pytesseract + pdf2image when MarkItDown returns insufficient text
- **Named entity recognition** — dslim/bert-base-NER extracts organizations and persons with two-tier confidence display (≥85% solid, 60–85% flagged)
- **Soft inference** — Meta-LLaMA 3.1 8B via Nebius AI (EU servers) infers seniority, transferable skills, career gaps, and languages
- **Dual extraction modes** — Hybrid (NER + LLaMA) or Full LLaMA, selectable per upload
- **Date normalization** — employment date spans normalized to MM-YYYY format with raw CV string preserved for transparency
- **Personality questionnaire** — a static decision tree (Sports, Reading, Handcraft & DIY, Writing, Music) maps candidate answers to trait scores (Teamwork, Leadership, Creativity, Discipline, Analytical, Problem Solving)

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Flask, Gunicorn |
| Text extraction | MarkItDown (Microsoft, local) |
| OCR | Tesseract, pdf2image, pytesseract |
| NER | `dslim/bert-base-NER` via HuggingFace Transformers |
| LLM inference | Meta-LLaMA 3.1 8B via Nebius AI (EU, GDPR DPA) |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Docker, DigitalOcean App Platform |

---

## Local Setup

**Prerequisites:** Python 3.11+, Tesseract (`brew install tesseract`), Poppler (`brew install poppler`)

```bash
git clone https://github.com/Eliakrl/lyra-cv-parser.git
cd lyra-cv-parser

python -m venv venv
source venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Create a `.env` file:

```
NEBIUS_API_KEY=your_key_here
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1
```

Run:

```bash
python app.py
```

Open `http://localhost:5000`.


## Architecture

```
CV upload (PDF / DOCX)          Personality questionnaire
        │                                │
        ▼                                ▼
MarkItDown (local)              Static decision tree
        │                       (hobby → follow-up → leaf)
        │ <20 chars?                     │
        ▼                                ▼
Tesseract OCR                   Trait score mapping
        │                       (Teamwork, Creativity…)
        ▼
DistilBERT NER
(orgs, persons)
        │
        ▼
LLaMA 3.1 8B — Nebius EU
(seniority, skills, gaps, languages)
        │
        ▼
Unified JSON profile
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `NEBIUS_API_KEY` | API key for Nebius AI |
| `NEBIUS_BASE_URL` | Nebius endpoint (default: `https://api.studio.nebius.ai/v1`) |

---

## Notes

- CV text is never stored or used for model training
- LLM inference routes through Nebius AI (Netherlands), keeping data within the EU
- The personality questionnaire is a prototype — trait mappings are approximations, not scientifically validated
