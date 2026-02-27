import csv
import re
from pathlib import Path

csv_path = Path(r"c:\Users\arikk\Desktop\תזה - תואר שני\אריאל קבצי עבודה זמניים\כל המשתתפים - CSV.csv")
out = Path("dataset_missing_pairs_report.csv")
missing = []
rows = 0

with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    rd = csv.reader(f)
    next(rd, None)
    for row in rd:
        if not row:
            continue
        rows += 1
        subj_raw = (row[0] if len(row) > 0 else "").strip()
        m_subj = re.search(r"\d+", subj_raw)
        if not m_subj:
            continue
        subj = m_subj.group(0)

        for q in range(1, 11):
            choice_idx = 7 + (q - 1) * 2
            text_idx = choice_idx + 1
            choice = (row[choice_idx] if choice_idx < len(row) else "").strip()
            text = (row[text_idx] if text_idx < len(row) else "").strip()

            m_ans = re.search(r"(\d+)", choice)
            ans = m_ans.group(1) if m_ans else ""

            if (not ans) or (text == ""):
                reason = []
                if not ans:
                    reason.append("missing_option")
                if text == "":
                    reason.append("missing_text")
                missing.append((subj, q, ans, ";".join(reason), choice, text))

with out.open("w", encoding="utf-8-sig", newline="") as w:
    w.write("subject,question,answer,reason,raw_option,raw_text\n")
    for subj, q, ans, reason, choice, text in missing:
        choice = choice.replace('"', '""')
        text = text.replace('"', '""')
        w.write(f'{subj},{q},{ans},{reason},"{choice}","{text}"\n')

print(f"ROWS={rows}")
print(f"MISSING={len(missing)}")
print(f"OUT={out.as_posix()}")
