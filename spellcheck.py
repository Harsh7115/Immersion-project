# spellcheck.py
from spellchecker import SpellChecker

spell = SpellChecker()
custom_vocab = set()

def load_custom_vocab(docs: list[str]):
    """Load custom vocab from project documents."""
    global custom_vocab
    for d in docs:
        for w in d.split():
            custom_vocab.add(w.lower())

def autocorrect_query(query: str) -> tuple[str, str]:
    """
    Autocorrects a query and returns (corrected_query, suggestion).
    Suggestion will be a 'Did you mean...' string or "" if no change.
    """
    words = query.split()
    corrected = []

    for w in words:
        candidates = spell.candidates(w)
        # Prefer candidates from custom vocab if available
        custom_candidates = [c for c in candidates if c in custom_vocab]
        if custom_candidates:
            corrected.append(custom_candidates[0])
        else:
            corrected.append(spell.correction(w) or w)

    corrected_query = " ".join(corrected)
    if corrected_query != query:
        return corrected_query, f"Did you mean '{corrected_query}'?"
    return query, ""

