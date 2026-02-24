import regex as re

# Hebrew diacritics (niqqud) and punctuation ranges
NIKKUD_RE = re.compile(r"[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]")
HEBREW_FINALS = {
    'כ': 'כ',
    'ך': 'כ',
    'מ': 'מ',
    'ם': 'מ',
    'נ': 'נ',
    'ן': 'נ',
    'פ': 'פ',
    'ף': 'פ',
    'צ': 'צ',
    'ץ': 'צ'
}

PUNCT_RE = re.compile(r"[\u200F\u200E\u202A-\u202E]|[\p{P}\p{S}]", flags=re.UNICODE)

# fallback punctuation removal (basic)
BASIC_PUNCT = re.compile(r"[\n\r\t]|")


def remove_niqqud(text: str) -> str:
    return NIKKUD_RE.sub('', text)


def normalize_finals(text: str) -> str:
    return ''.join(HEBREW_FINALS.get(ch, ch) for ch in text)


def normalize_hebrew(text: str) -> str:
    """Normalize Hebrew text: remove niqqud, map final forms to canonical, trim whitespace.

    This function intentionally stays light-weight (no heavy ML lemmatization).
    """
    if not isinstance(text, str):
        return text
    t = text.strip()
    t = remove_niqqud(t)
    t = normalize_finals(t)
    # replace multiple spaces with single
    t = re.sub(r"\s+", ' ', t)
    return t


if __name__ == '__main__':
    sample = 'מָקּוֹר\nתיאור\n'
    print(normalize_hebrew(sample))
