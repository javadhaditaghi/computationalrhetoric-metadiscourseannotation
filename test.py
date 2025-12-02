import pandas as pd
from unidecode import unidecode
import string
import re


# Copy the normalization functions here for standalone testing
def _normalize_expression(expr: str) -> str:
    """Normalize expression for boundary comparison"""
    if not expr or pd.isna(expr):
        return ""

    normalized = unidecode(expr)
    normalized = normalized.lower()
    normalized = normalized.strip(string.punctuation + string.whitespace)
    normalized = ' '.join(normalized.split())

    return normalized


def _normalize_classification(value: str) -> str:
    """Normalize classification values"""
    if not value or pd.isna(value):
        return ""

    normalized = unidecode(value)
    normalized = normalized.upper()
    normalized = normalized.strip(string.punctuation + string.whitespace)
    normalized = re.sub(r'[\s\-]+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    normalized = normalized.strip('_')

    return normalized


# TEST CASES
def test_expression_normalization():
    """Test expression normalization"""

    test_cases = [
        # (input1, input2, should_match, description)
        ("This thesis", "this thesis", True, "Case insensitive"),
        ("  This thesis  ", "This thesis", True, "Whitespace trimmed"),
        ("However,", "However", True, "Trailing comma removed"),
        (";However", "However", True, "Leading semicolon removed"),
        ("  , However ,  ", "however", True, "Complex punctuation + spaces"),
        ("café", "cafe", True, "Unicode accents"),
        ("naïve", "naive", True, "Unicode diaeresis"),
        ('"This thesis"', "This thesis", True, "Quotes stripped"),
        ("This  thesis", "This thesis", True, "Multiple spaces collapsed"),
        ("This thesis", "This", False, "Different content"),
        ("However", "Therefore", False, "Different words"),
    ]

    print("\n" + "=" * 80)
    print("EXPRESSION NORMALIZATION TESTS")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for expr1, expr2, should_match, description in test_cases:
        norm1 = _normalize_expression(expr1)
        norm2 = _normalize_expression(expr2)
        matches = (norm1 == norm2)

        if matches == should_match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        print(f"{status} | {description}")
        print(f"       Input 1: '{expr1}' → '{norm1}'")
        print(f"       Input 2: '{expr2}' → '{norm2}'")
        print(f"       Match: {matches} (Expected: {should_match})\n")

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80 + "\n")

    return failed == 0


def test_classification_normalization():
    """Test classification normalization"""

    test_cases = [
        # (input1, input2, should_match, description)
        ("METADISCOURSE", "metadiscourse", True, "Case insensitive"),
        ("Metadiscourse", "METADISCOURSE", True, "Mixed case"),
        ("MICRO-SCOPE", "MICRO_SCOPE", True, "Hyphen to underscore"),
        ("Micro Scope", "MICRO_SCOPE", True, "Space to underscore"),
        ("Frame Marker", "FRAME_MARKER", True, "Space to underscore"),
        ("frame-marker", "FRAME_MARKER", True, "Hyphen + lowercase"),
        ("  Frame Marker  ", "FRAME_MARKER", True, "Whitespace stripped"),
        ("Frame  Marker", "FRAME_MARKER", True, "Multiple spaces"),
        ("INTERACTIVE", "Interactive", True, "Case normalization"),
        ("METADISCOURSE", "PROPOSITIONAL", False, "Different values"),
    ]

    print("\n" + "=" * 80)
    print("CLASSIFICATION NORMALIZATION TESTS")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for val1, val2, should_match, description in test_cases:
        norm1 = _normalize_classification(val1)
        norm2 = _normalize_classification(val2)
        matches = (norm1 == norm2)

        if matches == should_match:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        print(f"{status} | {description}")
        print(f"       Input 1: '{val1}' → '{norm1}'")
        print(f"       Input 2: '{val2}' → '{norm2}'")
        print(f"       Match: {matches} (Expected: {should_match})\n")

    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80 + "\n")

    return failed == 0


def test_edge_cases():
    """Test edge cases"""

    print("\n" + "=" * 80)
    print("EDGE CASE TESTS")
    print("=" * 80 + "\n")

    edge_cases = [
        (None, "None/NaN handling"),
        ("", "Empty string"),
        ("   ", "Only whitespace"),
        ("...", "Only punctuation"),
        ("   ,,,   ", "Punctuation + whitespace"),
    ]

    passed = 0

    for expr, description in edge_cases:
        try:
            result = _normalize_expression(expr)
            print(f"✅ PASS | {description}")
            print(f"       Input: {repr(expr)} → Output: '{result}'\n")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL | {description}")
            print(f"       Input: {repr(expr)}")
            print(f"       Error: {e}\n")

    print("=" * 80)
    print(f"RESULTS: {passed} passed out of {len(edge_cases)} edge cases")
    print("=" * 80 + "\n")

    return passed == len(edge_cases)


if __name__ == "__main__":
    print("\n" + "🧪 STARTING NORMALIZATION TESTS" + "\n")

    expr_pass = test_expression_normalization()
    class_pass = test_classification_normalization()
    edge_pass = test_edge_cases()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    if expr_pass and class_pass and edge_pass:
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        if not expr_pass:
            print("   - Expression normalization tests failed")
        if not class_pass:
            print("   - Classification normalization tests failed")
        if not edge_pass:
            print("   - Edge case tests failed")

    print("=" * 80 + "\n")