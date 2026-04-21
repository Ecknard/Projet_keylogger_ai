"""
tests/test_all.py — Tests unitaires
TP1 — Intelligence Artificielle & Cybersécurité

Couverture :
    - sentiment_analyzer   : analyze_sentiment, cas limites
    - sensitive_detector   : regex patterns, masquage
    - anomaly_detector     : extract_features
    - encryption (optionnel si cryptography installé)

Exécution : python -m pytest tests/ -v
         ou python tests/test_all.py
"""

import sys
import os
import unittest

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Tests — sentiment_analyzer
# ---------------------------------------------------------------------------
class TestSentimentAnalyzer(unittest.TestCase):

    def setUp(self):
        from sentiment_analyzer import analyze_sentiment
        self.analyze = analyze_sentiment

    def test_positive_text(self):
        result = self.analyze("I am absolutely happy and delighted today!")
        self.assertEqual(result["label"], "positif")
        self.assertGreater(result["score"], 0.05)

    def test_negative_text(self):
        result = self.analyze("I hate this terrible awful broken system so much!")
        self.assertEqual(result["label"], "négatif")
        self.assertLess(result["score"], -0.05)

    def test_neutral_text(self):
        result = self.analyze("The file was saved to the directory.")
        self.assertIn(result["label"], ["neutre", "positif", "négatif"])  # dépend du modèle

    def test_too_short(self):
        result = self.analyze("Hi")
        self.assertEqual(result["label"], "trop_court")
        self.assertEqual(result["score"], 0.0)

    def test_empty_string(self):
        result = self.analyze("")
        self.assertEqual(result["label"], "trop_court")

    def test_return_structure(self):
        result = self.analyze("This is a test sentence for structure validation.")
        self.assertIn("score", result)
        self.assertIn("label", result)
        self.assertIn("timestamp", result)
        self.assertIn("text", result)

    def test_score_range(self):
        result = self.analyze("Testing the score range for this text.")
        self.assertGreaterEqual(result["score"], -1.0)
        self.assertLessEqual(result["score"], 1.0)


# ---------------------------------------------------------------------------
# Tests — sensitive_detector
# ---------------------------------------------------------------------------
class TestSensitiveDetector(unittest.TestCase):

    def setUp(self):
        from sensitive_detector import detect_with_regex, mask_sensitive, compute_entropy
        self.detect   = detect_with_regex
        self.mask     = mask_sensitive
        self.entropy  = compute_entropy

    def test_detect_email(self):
        text = "Contact me at alice@example.com please"
        detections = self.detect(text)
        types = [d["type"] for d in detections]
        self.assertIn("email", types)

    def test_detect_credit_card(self):
        text = "My card number is 4532 1234 5678 9012"
        detections = self.detect(text)
        types = [d["type"] for d in detections]
        self.assertIn("carte_bancaire", types)

    def test_detect_french_phone(self):
        text = "Appelez-moi au 06 12 34 56 78 svp"
        detections = self.detect(text)
        types = [d["type"] for d in detections]
        self.assertIn("telephone_fr", types)

    def test_detect_secu(self):
        text = "Mon numéro : 1 85 12 75 123 456 78"
        detections = self.detect(text)
        types = [d["type"] for d in detections]
        self.assertIn("numero_secu_fr", types)

    def test_no_false_positive_plain_text(self):
        text = "Bonjour, aujourd'hui nous allons travailler sur le projet Python."
        detections = self.detect(text)
        # Texte ordinaire ne doit pas être détecté
        self.assertEqual(len(detections), 0)

    def test_mask_replaces_correctly(self):
        text = "Email: alice@example.com here"
        detections = self.detect(text)
        masked = self.mask(text, detections)
        self.assertNotIn("alice@example.com", masked)
        self.assertIn("*", masked)

    def test_mask_length_preserved(self):
        """Le texte masqué doit avoir la même longueur que l'original."""
        text = "Card: 4532 1234 5678 9012 end"
        detections = self.detect(text)
        masked = self.mask(text, detections)
        self.assertEqual(len(text), len(masked))

    def test_entropy_empty(self):
        self.assertEqual(self.entropy(""), 0.0)

    def test_entropy_uniform(self):
        # "aaa" → entropie très faible
        self.assertLess(self.entropy("aaa"), 0.5)

    def test_entropy_varied(self):
        # "aAbBcC123!" → entropie élevée
        self.assertGreater(self.entropy("aAbBcC123!"), 3.0)


# ---------------------------------------------------------------------------
# Tests — anomaly_detector
# ---------------------------------------------------------------------------
class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):
        from anomaly_detector import extract_features
        self.extract = extract_features

    def _make_window(self, n=20, delay=0.12, key_type="alphanum"):
        import time
        return [
            {"timestamp": time.time() + i * delay,
             "inter_key_delay": delay,
             "key_type": key_type}
            for i in range(n)
        ]

    def test_extract_returns_array(self):
        import numpy as np
        window = self._make_window()
        features = self.extract(window)
        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (1, 8))

    def test_extract_too_small_window(self):
        features = self.extract([{"inter_key_delay": 0.1, "key_type": "alphanum"}])
        self.assertIsNone(features)

    def test_extract_empty(self):
        features = self.extract([])
        self.assertIsNone(features)

    def test_fast_typing_has_low_mean_delay(self):
        import numpy as np
        fast_window = self._make_window(delay=0.05)
        features = self.extract(fast_window)
        self.assertIsNotNone(features)
        mean_delay = features[0][0]
        self.assertLess(mean_delay, 0.1)

    def test_slow_typing_has_high_mean_delay(self):
        slow_window = self._make_window(delay=1.5)
        features = self.extract(slow_window)
        self.assertIsNotNone(features)
        mean_delay = features[0][0]
        self.assertGreater(mean_delay, 1.0)


# ---------------------------------------------------------------------------
# Tests — encryption (optionnel)
# ---------------------------------------------------------------------------
class TestEncryption(unittest.TestCase):

    def setUp(self):
        try:
            from extension.encryption import (
                generate_key, encrypt_text, decrypt_text, derive_key_from_password
            )
            self.generate_key  = generate_key
            self.encrypt       = encrypt_text
            self.decrypt       = decrypt_text
            self.derive        = derive_key_from_password
            self._available    = True
        except ImportError:
            self._available = False

    def test_encrypt_decrypt_roundtrip(self):
        if not self._available:
            self.skipTest("cryptography non installé")
        key = self.generate_key()
        plaintext = "Secret message: P@ssw0rd123!"
        encrypted = self.encrypt(plaintext, key)
        decrypted = self.decrypt(encrypted, key)
        self.assertEqual(plaintext, decrypted)

    def test_different_nonces_each_time(self):
        if not self._available:
            self.skipTest("cryptography non installé")
        key = self.generate_key()
        enc1 = self.encrypt("same text", key)
        enc2 = self.encrypt("same text", key)
        # AES-GCM utilise un nonce aléatoire → deux chiffrés différents
        self.assertNotEqual(enc1, enc2)

    def test_wrong_key_raises(self):
        if not self._available:
            self.skipTest("cryptography non installé")
        from cryptography.exceptions import InvalidTag
        key1 = self.generate_key()
        key2 = self.generate_key()
        encrypted = self.encrypt("test", key1)
        with self.assertRaises(Exception):
            self.decrypt(encrypted, key2)

    def test_pbkdf2_deterministic(self):
        if not self._available:
            self.skipTest("cryptography non installé")
        password = "MonMotDePasse42!"
        key1, salt = self.derive(password)
        key2, _    = self.derive(password, salt)
        self.assertEqual(key1, key2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Tests unitaires — TP1 AI-Driven Malware")
    print("=" * 60)
    unittest.main(verbosity=2)
