"""Line post-processing: the "safe" mode of scripts/vietocr_npu_pipeline.post_process_vietnamese
(no rules derived from the evaluation pages)."""

import re
import unicodedata


def normalize_text(text):
    s = unicodedata.normalize("NFC", text)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return " ".join(s.split())


def post_process(text):
    s = text
    s = re.sub(r'(?<![\w@])([a-z0-9._%+-]+)O([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,})(?![\w@])', r'\1@\2', s)
    s = re.sub(r'Tr[âậ]n trọng\s*\.\s*\]', 'Trân trọng./.', s)
    s = re.sub(r'Tr[âậ]n trọng\s*\]', 'Trân trọng./.', s)
    s = re.sub(r'Trân trọng\.\s*4\s*\.?$', 'Trân trọng./.', s)
    s = re.sub(r'Độc lập\s+i\s+Tự do', 'Độc lập - Tự do', s)
    s = re.sub(r'Tự do\s+(?:7|i)\s+Hạnh phúc', 'Tự do - Hạnh phúc', s)
    s = s.replace("Bộ Iuật", "Bộ luật").replace("Iuật", "luật")
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return normalize_text(s)
