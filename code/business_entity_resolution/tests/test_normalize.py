"""Tests for conservative text normalization utilities."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize import (address_tokens, core_name, core_name_tokens, extract_numeric_tokens, name_search_text, name_tokens, normalize_address, normalize_country, normalize_name, normalize_text, postal_candidate, postal_candidates, standardize_address, strip_latin_accents)
class NormalizeTests(unittest.TestCase):
    def test_case_whitespace_nfkc(self):
        self.assertEqual(normalize_text("  Ａcme   CORP  "), "acme corp")
        self.assertEqual(normalize_name("  ACME   CORP. "), "acme corp")
    def test_punctuation_ampersand_apostrophe(self):
        self.assertEqual(normalize_name("Bonilla’s & Sons, Inc."), "bonillas and sons inc")
        self.assertEqual(normalize_name("A.B.C. Technologies"), "a b c technologies")
    def test_ordered_tokens_retain_repetition(self):
        self.assertEqual(name_tokens("ABC Technologies Technologies"), ("abc", "technologies", "technologies"))
        self.assertEqual(name_tokens("ABC Technologies Pvt Ltd", core=True), ("abc", "technologies"))
    def test_suffixes_terminal_and_basic_preserved(self):
        cases = {"TATA Consultancy Services Ltd.": "tata consultancy services", "Acme Private Limited": "acme", "Acme Pvt Ltd": "acme", "Acme Incorporated": "acme", "Acme Corp.": "acme", "Acme L.L.C.": "acme", "Acme GmbH": "acme", "Acme SAS": "acme"}
        for original, expected in cases.items():
            with self.subTest(original=original):
                self.assertEqual(core_name(original), expected)
                self.assertNotEqual(normalize_name(original), expected)
    def test_suffix_middle_and_location_tokens_preserved(self):
        self.assertEqual(core_name("Limited Brands International"), "limited brands international")
        self.assertEqual(core_name("Private Company Holdings"), "private company holdings")
        self.assertEqual(core_name("ABC Technologies India Pvt Ltd"), "abc technologies india")
        self.assertEqual(core_name("ABC Technologies USA Pvt Ltd"), "abc technologies usa")
        self.assertNotEqual(core_name("ABC Technologies India Pvt Ltd"), core_name("ABC Technologies USA Pvt Ltd"))
    def test_unicode_accents_and_search(self):
        self.assertEqual(normalize_name("Café Société"), "café société")
        self.assertEqual(strip_latin_accents("Café"), "Cafe")
        self.assertEqual(name_search_text("Café Société"), "cafe societe")
        self.assertTrue(normalize_name("東京株式会社"))
    def test_indic_combining_marks_preserved(self):
        self.assertEqual(normalize_name("सेवन एनर्जी प्राइवेट लिमिटेड"), "सेवन एनर्जी प्राइवेट लिमिटेड")
        self.assertEqual(name_tokens("स्वस्तिक केयर"), ("स्वस्तिक", "केयर"))
        self.assertEqual(normalize_address("हाउस नं. १२"), "हाउस नं १२")
    def test_missing_values(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(normalize_name(value), "")
                self.assertEqual(core_name(value), "")
                self.assertEqual(name_tokens(value), ())
                self.assertEqual(normalize_address(value), "")
                self.assertEqual(address_tokens(value), ())
                self.assertEqual(extract_numeric_tokens(value), ())
                self.assertIsNone(postal_candidate(value, "France"))
    def test_address_punctuation(self):
        self.assertEqual(normalize_address("  12, MG Road. Bengaluru  "), "12 mg road bengaluru")
        self.assertEqual(address_tokens("12 MG Road"), ("12", "mg", "road"))
    def test_separate_conservative_address_standardization(self):
        self.assertEqual(normalize_address("12 MG Rd, Suite 4"), "12 mg rd suite 4")
        self.assertEqual(standardize_address("12 MG Rd, Suite 4"), "12 mg road suite 4")
        self.assertEqual(standardize_address("9 Main St, 4th Ave"), "9 main st 4th avenue")
        self.assertEqual(standardize_address("5 Oak Dr"), "5 oak drive")
        self.assertEqual(standardize_address("7 Cedar Ln"), "7 cedar lane")
        self.assertEqual(standardize_address("8 Hill Trl"), "8 hill trail")
        self.assertEqual(standardize_address("2 Lake Cir"), "2 lake circle")
    def test_numbers_preserved_in_order_and_leading_zeros(self):
        self.assertEqual(extract_numeric_tokens("12 MG Road, Bengaluru 00560001, Apt 12B"), ("12", "00560001", "12"))
        self.assertNotEqual(normalize_address("12 MG Road"), normalize_address("92 MG Road"))
        self.assertNotEqual(extract_numeric_tokens("12 MG Road"), extract_numeric_tokens("92 MG Road"))
    def test_country_postal_heuristics(self):
        self.assertEqual(postal_candidates("Main St, 02108-1234", "US"), ("02108-1234",))
        self.assertEqual(postal_candidates("Bengaluru 560001", "India"), ("560001",))
        self.assertEqual(postal_candidates("Paris 75001", "France"), ("75001",))
        self.assertIsNone(postal_candidate("Paris 75001", "Francee"))
        self.assertEqual(postal_candidates("Paris 75001", "unknown country"), ())
        self.assertEqual(postal_candidates("PIN 012345", "India"), ())
    def test_open_set_country_normalization(self):
        self.assertEqual(normalize_country("  FＲＡＮＣＥ "), "france")
        self.assertEqual(normalize_country("  Côte d’Ivoire "), "côte d’ivoire")
        self.assertEqual(normalize_country(" Atlantis "), "atlantis")
        self.assertEqual(normalize_country(None), "")
    def test_deterministic_and_core_tokens(self):
        self.assertEqual(core_name_tokens("Acme LLC"), ("acme",))
        self.assertEqual(normalize_name("L’École & Sons, Pvt. Ltd."), normalize_name("L’École & Sons, Pvt. Ltd."))
        self.assertEqual(normalize_address("12 Rue de l’Église"), normalize_address("12 Rue de l’Église"))
if __name__ == "__main__":
    unittest.main()
