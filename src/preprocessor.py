"""Text Preprocessing and NLP Cleaning module for HireSense AI.

Handles text sanitization, tokenization, stopword removal, and normalization
while preserving domain-specific technical tokens (e.g., C++, .NET, CI/CD).
"""

from __future__ import annotations

import re
from typing import List, Set


# Standard curated English stopwords list (avoids runtime NLTK download dependencies)
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any",
    "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did",
    "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each",
    "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's",
    "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves",
    # Resume filler words
    "candidate", "curriculum", "vitae", "resume", "page", "phone", "email",
    "responsible", "experienced", "experience", "duties", "etc", "years",
}

# Special technical tokens mapping to preserve them during punctuation stripping
TECH_TOKEN_REPLACEMENTS = {
    r"(?i)(?:\b|(?<=\s))c\+\+(?=[,\s\.;:\)]|$)": " cplusplus ",
    r"(?i)(?:\b|(?<=\s))c\#(?=[,\s\.;:\)]|$)": " csharp ",
    r"(?i)(?:\b|(?<=\s))\.net(?=[,\s\.;:\)]|$)": " dotnet ",
    r"(?i)\bnode\.js\b": " nodejs ",
    r"(?i)\bnext\.js\b": " nextjs ",
    r"(?i)\bvue\.js\b": " vuejs ",
    r"(?i)\breact\.js\b": " react ",
    r"(?i)\bci[\/\\]cd\b": " cicd ",
    r"(?i)\btcp\/ip\b": " tcpip ",
}


def clean_text_for_nlp(text: str, remove_stopwords: bool = True) -> str:
    """Cleans and normalizes text for NLP and TF-IDF comparison.

    Steps:
    1. Lowercase conversion.
    2. Remove URLs, email addresses, and phone numbers to prevent demographic/contact bias.
    3. Normalize bullet symbols and whitespace.
    4. Preserve special technical terms (c++, c#, .net, node.js, ci/cd).
    5. Strip non-alphanumeric punctuation.
    6. Optionally filter out common English and resume filler stopwords.
    """
    if not text:
        return ""

    cleaned = text.lower()

    # 1. Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # 2. Strip emails
    cleaned = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b", " ", cleaned)

    # 3. Strip phone numbers
    cleaned = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", " ", cleaned)

    # 4. Normalize bullets and special Unicode spaces
    cleaned = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219\uf0b7\*\-—–|]", " ", cleaned)

    # 5. Protect tech tokens
    for pattern, replacement in TECH_TOKEN_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # 6. Replace non-alphanumeric chars (except whitespace) with spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # 7. Normalize whitespace
    words = cleaned.split()

    # 8. Filter stopwords if requested
    if remove_stopwords:
        words = [w for w in words if w not in STOPWORDS and len(w) > 1]

    return " ".join(words)


def extract_keywords_simple(text: str, top_n: int = 15) -> List[str]:
    """Extract top frequent significant terms from text (heuristic word frequency)."""
    cleaned = clean_text_for_nlp(text, remove_stopwords=True)
    words = cleaned.split()
    counts: dict[str, int] = {}
    for word in words:
        if len(word) >= 3:
            counts[word] = counts.get(word, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_items[:top_n]]
