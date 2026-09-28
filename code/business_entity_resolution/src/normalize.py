import os
import re
import unicodedata
from typing import List, Tuple, Dict, Optional
import pandas as pd

# Compilation of regular expressions for performance
RE_PUNCT_NOISE = re.compile(r'[\[\]\(\)\{\}\<\>\#\*\+\=\_\~\|\^\`\:\;\"\'\\]')
RE_AMPERSAND = re.compile(r'\s*&\s*')
RE_MULTISPACE = re.compile(r'\s+')
RE_NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')

# Common legal and corporate abbreviations mapping
LEGAL_MAPPINGS = [
    (re.compile(r'\b(private\s+limited|pvt\s+ltd|p\s+ltd|pvt\s+limited|private\s+ltd)\b', re.IGNORECASE), 'pvt ltd'),
    (re.compile(r'\b(corporation|corp\.)\b', re.IGNORECASE), 'corp'),
    (re.compile(r'\b(incorporated|inc\.)\b', re.IGNORECASE), 'inc'),
    (re.compile(r'\b(limited|ltd\.)\b', re.IGNORECASE), 'ltd'),
    (re.compile(r'\b(l\.l\.c\.|l\s+l\s+c|llc\.)\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\b(l\.l\.p\.|llp\.)\b', re.IGNORECASE), 'llp'),
    (re.compile(r'\b(company|co\.)\b', re.IGNORECASE), 'co'),
    # French corporate legal suffixes (supporting with or without dots and spaces)
    (re.compile(r'\b(s\s*a\s*r\s*l|sarl)\b', re.IGNORECASE), 'sarl'),
    (re.compile(r'\b(s\s*a\s*s\s*u|sasu)\b', re.IGNORECASE), 'sasu'),
    (re.compile(r'\b(s\s*a\s*s|sas)\b', re.IGNORECASE), 'sas'),
    (re.compile(r'\b(e\s*u\s*r\s*l|eurl)\b', re.IGNORECASE), 'eurl'),
    (re.compile(r'\b(s\s*c\s*i|sci)\b', re.IGNORECASE), 'sci'),
    (re.compile(r'\b(s\s*n\s*c|snc)\b', re.IGNORECASE), 'snc'),
    (re.compile(r'\b(s\s*e\s*l\s*a\s*r\s*l|selarl)\b', re.IGNORECASE), 'selarl'),
    (re.compile(r'\b(s\.a\.|s\s+a)\b', re.IGNORECASE), 'sa'),
    (re.compile(r'\b(gmbh)\b', re.IGNORECASE), 'gmbh'),
]

# Common address abbreviations mapping
ADDRESS_MAPPINGS = [
    (re.compile(r'\b(street|str\.)\b', re.IGNORECASE), 'st'),
    (re.compile(r'\b(road|rd\.)\b', re.IGNORECASE), 'rd'),
    (re.compile(r'\b(avenue|ave\.|ave|av\.|av)\b', re.IGNORECASE), 'ave'),
    (re.compile(r'\b(drive|dr\.)\b', re.IGNORECASE), 'dr'),
    (re.compile(r'\b(boulevard|blvd\.|blvd|bd\.|bd|bld)\b', re.IGNORECASE), 'blvd'),
    (re.compile(r'\b(lane|ln\.)\b', re.IGNORECASE), 'ln'),
    (re.compile(r'\b(court|ct\.)\b', re.IGNORECASE), 'ct'),
    (re.compile(r'\b(circle|cir\.)\b', re.IGNORECASE), 'cir'),
    (re.compile(r'\b(place|pl\.)\b', re.IGNORECASE), 'pl'),
    (re.compile(r'\b(suite|ste\.)\b', re.IGNORECASE), 'ste'),
    (re.compile(r'\b(apartment|apt\.)\b', re.IGNORECASE), 'apt'),
    (re.compile(r'\b(floor|flr\.|fl\.)\b', re.IGNORECASE), 'fl'),
    (re.compile(r'\b(building|bldg\.)\b', re.IGNORECASE), 'bldg'),
    (re.compile(r'\b(highway|hwy\.)\b', re.IGNORECASE), 'hwy'),
    (re.compile(r'\b(parkway|pkwy\.)\b', re.IGNORECASE), 'pkwy'),
    (re.compile(r'\b(north)\b', re.IGNORECASE), 'n'),
    (re.compile(r'\b(south)\b', re.IGNORECASE), 's'),
    (re.compile(r'\b(east)\b', re.IGNORECASE), 'e'),
    (re.compile(r'\b(west)\b', re.IGNORECASE), 'w'),
    (re.compile(r'\b(northeast|ne\.)\b', re.IGNORECASE), 'ne'),
    (re.compile(r'\b(northwest|nw\.)\b', re.IGNORECASE), 'nw'),
    (re.compile(r'\b(southeast|se\.)\b', re.IGNORECASE), 'se'),
    (re.compile(r'\b(southwest|sw\.)\b', re.IGNORECASE), 'sw'),
    # French street/address standards (global normalization, no country branching)
    (re.compile(r'\b(rue)\b', re.IGNORECASE), 'r'),
    (re.compile(r'\b(chemin|ch\.)\b', re.IGNORECASE), 'chemin'),
    (re.compile(r'\b(impasse|imp\.)\b', re.IGNORECASE), 'impasse'),
    (re.compile(r'\b(allee|all\.)\b', re.IGNORECASE), 'allee'),
    (re.compile(r'\b(cours|crs\.)\b', re.IGNORECASE), 'cours'),
    (re.compile(r'\b(route|rte\.)\b', re.IGNORECASE), 'rte'),
    (re.compile(r'\b(quai)\b', re.IGNORECASE), 'quai'),
    (re.compile(r'\b(bis)\b', re.IGNORECASE), 'b'),
    (re.compile(r'\b(ter)\b', re.IGNORECASE), 'c'),
]

# US state names to 2-letter codes for normalization
US_STATE_MAPPINGS = [
    (re.compile(r'\b(alabama)\b', re.IGNORECASE), 'al'),
    (re.compile(r'\b(alaska)\b', re.IGNORECASE), 'ak'),
    (re.compile(r'\b(arizona)\b', re.IGNORECASE), 'az'),
    (re.compile(r'\b(arkansas)\b', re.IGNORECASE), 'ar'),
    (re.compile(r'\b(california)\b', re.IGNORECASE), 'ca'),
    (re.compile(r'\b(colorado)\b', re.IGNORECASE), 'co'),
    (re.compile(r'\b(connecticut)\b', re.IGNORECASE), 'ct'),
    (re.compile(r'\b(delaware)\b', re.IGNORECASE), 'de'),
    (re.compile(r'\b(florida)\b', re.IGNORECASE), 'fl'),
    (re.compile(r'\b(georgia)\b', re.IGNORECASE), 'ga'),
    (re.compile(r'\b(hawaii)\b', re.IGNORECASE), 'hi'),
    (re.compile(r'\b(idaho)\b', re.IGNORECASE), 'id'),
    (re.compile(r'\b(illinois)\b', re.IGNORECASE), 'il'),
    (re.compile(r'\b(indiana)\b', re.IGNORECASE), 'in'),
    (re.compile(r'\b(iowa)\b', re.IGNORECASE), 'ia'),
    (re.compile(r'\b(kansas)\b', re.IGNORECASE), 'ks'),
    (re.compile(r'\b(kentucky)\b', re.IGNORECASE), 'ky'),
    (re.compile(r'\b(louisiana)\b', re.IGNORECASE), 'la'),
    (re.compile(r'\b(maine)\b', re.IGNORECASE), 'me'),
    (re.compile(r'\b(maryland)\b', re.IGNORECASE), 'md'),
    (re.compile(r'\b(massachusetts)\b', re.IGNORECASE), 'ma'),
    (re.compile(r'\b(michigan)\b', re.IGNORECASE), 'mi'),
    (re.compile(r'\b(minnesota)\b', re.IGNORECASE), 'mn'),
    (re.compile(r'\b(mississippi)\b', re.IGNORECASE), 'ms'),
    (re.compile(r'\b(missouri)\b', re.IGNORECASE), 'mo'),
    (re.compile(r'\b(montana)\b', re.IGNORECASE), 'mt'),
    (re.compile(r'\b(nebraska)\b', re.IGNORECASE), 'ne'),
    (re.compile(r'\b(nevada)\b', re.IGNORECASE), 'nv'),
    (re.compile(r'\b(new hampshire)\b', re.IGNORECASE), 'nh'),
    (re.compile(r'\b(new jersey)\b', re.IGNORECASE), 'nj'),
    (re.compile(r'\b(new mexico)\b', re.IGNORECASE), 'nm'),
    (re.compile(r'\b(new york)\b', re.IGNORECASE), 'ny'),
    (re.compile(r'\b(north carolina)\b', re.IGNORECASE), 'nc'),
    (re.compile(r'\b(north dakota)\b', re.IGNORECASE), 'nd'),
    (re.compile(r'\b(ohio)\b', re.IGNORECASE), 'oh'),
    (re.compile(r'\b(oklahoma)\b', re.IGNORECASE), 'ok'),
    (re.compile(r'\b(oregon)\b', re.IGNORECASE), 'or'),
    (re.compile(r'\b(pennsylvania)\b', re.IGNORECASE), 'pa'),
    (re.compile(r'\b(rhode island)\b', re.IGNORECASE), 'ri'),
    (re.compile(r'\b(south carolina)\b', re.IGNORECASE), 'sc'),
    (re.compile(r'\b(south dakota)\b', re.IGNORECASE), 'sd'),
    (re.compile(r'\b(tennessee)\b', re.IGNORECASE), 'tn'),
    (re.compile(r'\b(texas)\b', re.IGNORECASE), 'tx'),
    (re.compile(r'\b(utah)\b', re.IGNORECASE), 'ut'),
    (re.compile(r'\b(vermont)\b', re.IGNORECASE), 'vt'),
    (re.compile(r'\b(virginia)\b', re.IGNORECASE), 'va'),
    (re.compile(r'\b(washington)\b', re.IGNORECASE), 'wa'),
    (re.compile(r'\b(west virginia)\b', re.IGNORECASE), 'wv'),
    (re.compile(r'\b(wisconsin)\b', re.IGNORECASE), 'wi'),
    (re.compile(r'\b(wyoming)\b', re.IGNORECASE), 'wy'),
]

def strip_accents(text: str) -> str:
    """Normalize unicode characters and strip accents while preserving characters."""
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text_basic(text: str) -> str:
    """Basic lowercasing, accent stripping, ampersand normalization, and noise stripping."""
    if text is None or not isinstance(text, str):
        return ""
    t = text.strip()
    if t.lower() in ('null', 'none', 'nan'):
        return ""
    t = strip_accents(t.lower())
    t = RE_AMPERSAND.sub(' and ', t)
    t = RE_PUNCT_NOISE.sub(' ', t)
    t = re.sub(r'[,.\-\/]', ' ', t)
    t = RE_MULTISPACE.sub(' ', t).strip()
    return t

def normalize_name(name: str) -> str:
    """Normalize business name: expand/standardize legal suffixes, remove punctuation noise."""
    t = clean_text_basic(name)
    if not t:
        return ""
    # Standardize legal suffixes
    for pattern, repl in LEGAL_MAPPINGS:
        t = pattern.sub(repl, t)
    t = RE_NON_ALPHANUM.sub(' ', t)
    t = RE_MULTISPACE.sub(' ', t).strip()
    return t

def normalize_address(address: str) -> str:
    """Normalize business address: standardize road/street words, state abbreviations."""
    t = clean_text_basic(address)
    if not t:
        return ""
    # Standardize address terms
    for pattern, repl in ADDRESS_MAPPINGS:
        t = pattern.sub(repl, t)
    for pattern, repl in US_STATE_MAPPINGS:
        t = pattern.sub(repl, t)
    t = RE_NON_ALPHANUM.sub(' ', t)
    t = RE_MULTISPACE.sub(' ', t).strip()
    return t

def create_composite_representation(name: str, address: str) -> str:
    """Create composite string for blocking and candidate generation."""
    norm_n = normalize_name(name)
    norm_a = normalize_address(address)
    if norm_n and norm_a:
        # Give name extra weight in composite text by duplicating name tokens
        return f"{norm_n} {norm_n} {norm_a}"
    elif norm_n:
        return f"{norm_n} {norm_n}"
    elif norm_a:
        return norm_a
    return ""


def fast_clean_series(s: pd.Series) -> pd.Series:
    """
    Ultra-fast C-vectorized cleaning for pandas series:
    lowercasing, noise removal, and whitespace standardization.
    """
    s = s.fillna('').astype(str).str.lower()
    s = s.str.replace(r'[\[\]\(\)\{\}\<\>\#\*\+\=\_\~\|\^\`\:\;\"\'\\]', ' ', regex=True)
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace(r'[,.\-\/]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s


def build_vectorized_composite_corpus(names: pd.Series, addrs: pd.Series) -> List[str]:
    """
    Builds composite representations for millions of records using vectorized ops.
    Applies ADDRESS_MAPPINGS + LEGAL_MAPPINGS so that variants like:
      "Main Street" == "Main St", "Inc." == "Incorporated", "Pvt Ltd" == "Private Limited"
    all produce matching tokens, dramatically improving blocking recall.

    Weights business name 2x to give it more IDF influence.
    """
    clean_n = fast_clean_series(names)
    clean_a = fast_clean_series(addrs)

    # Normalize legal suffixes in names (e.g. "incorporated" -> "inc", "l.l.c." -> "llc")
    for pattern, repl in LEGAL_MAPPINGS:
        clean_n = clean_n.str.replace(pattern, repl, regex=True)

    # Normalize address abbreviations (e.g. "street" -> "st", "avenue" -> "ave")
    for pattern, repl in ADDRESS_MAPPINGS:
        clean_a = clean_a.str.replace(pattern, repl, regex=True)

    composite = clean_n + " " + clean_n + " " + clean_a
    return composite.tolist()


if __name__ == '__main__':
    # Unit tests on sample cases from our EDA
    test_cases = [
        ("Orelee's Barbershop", "1795 Westchester Drive, High Point, NC"),
        ("American Choice Service LLC", "216 Metropolitan Drive, NY, Rochester"),
        ("AMERICAN CHOICE SERVICE [[LLC]]", "1 METROPOLITAN DR, ROCHESTER, NY"),
        ("D+ Reit, Inc", "8 Forman Road, Currie, MN"),
        ("D+ Reit,", "8 Forman Rd, Curie, Minnesota"),
        ("White Infrastructure Pvt Ltd", "Office-No-207, Marathon Monte Plaza, Mumbai, Maharashtra"),
        ("Thermal & Fils SASU", "20 Rue Parmentier, Dunkerque, Hauts-de-France"),
    ]
    print("Testing Normalization on Sample Records:")
    for name, addr in test_cases:
        norm_name = normalize_name(name)
        norm_addr = normalize_address(addr)
        composite = create_composite_representation(name, addr)
        print(f"Original: Name='{name}' | Addr='{addr}'")
        print(f"  Norm Name: '{norm_name}'")
        print(f"  Norm Addr: '{norm_addr}'")
        print(f"  Composite: '{composite}'")
        print("-" * 50)
