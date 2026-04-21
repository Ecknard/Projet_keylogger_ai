"""
extension/encryption.py — Extension C : Chiffrement de bout en bout du fichier log
TP1 — Intelligence Artificielle & Cybersécurité

Mécanisme : AES-256-GCM (symétrique authentifié)
Bibliothèque : cryptography (pip install cryptography)

Avantages d'AES-GCM vs autres modes :
- GCM (Galois/Counter Mode) = chiffrement + authentification intégrée
- Détecte toute modification du texte chiffré (intégrité garantie)
- Plus sûr qu'AES-CBC/ECB qui n'ont pas d'authentification

Usage :
    key = generate_key()
    save_key(key, "data/secret.key")
    encrypted = encrypt_text("Hello World", key)
    decrypted = decrypt_text(encrypted, key)
"""

import base64
import os
from datetime import datetime

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False
    print("[AVERTISSEMENT] cryptography non installé : pip install cryptography")

KEY_SIZE    = 32   # 256 bits
NONCE_SIZE  = 12   # 96 bits (recommandé pour GCM)
SALT_SIZE   = 16


# ---------------------------------------------------------------------------
# Gestion des clés
# ---------------------------------------------------------------------------

def generate_key() -> bytes:
    """Génère une clé AES-256 aléatoire cryptographiquement sûre."""
    return os.urandom(KEY_SIZE)


def derive_key_from_password(password: str, salt: bytes = None) -> tuple:
    """
    Dérive une clé AES-256 à partir d'un mot de passe via PBKDF2-SHA256.

    Retour
    ------
    (key: bytes, salt: bytes)
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography non disponible.")

    if salt is None:
        salt = os.urandom(SALT_SIZE)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt,
        iterations=480_000,  # OWASP 2023 recommandation
    )
    key = kdf.derive(password.encode("utf-8"))
    return key, salt


def save_key(key: bytes, path: str = "data/secret.key") -> None:
    """Sauvegarde la clé en base64 dans un fichier protégé."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(base64.b64encode(key).decode("utf-8"))
    # Permissions restrictives (Unix)
    try:
        os.chmod(path, 0o600)
    except AttributeError:
        pass  # Windows ne supporte pas chmod
    print(f"[INFO] Clé sauvegardée → {path} (GARDEZ CE FICHIER SECRET !)")


def load_key(path: str = "data/secret.key") -> bytes:
    """Charge la clé depuis le fichier."""
    with open(path, "r") as f:
        return base64.b64decode(f.read().strip())


# ---------------------------------------------------------------------------
# Chiffrement / Déchiffrement
# ---------------------------------------------------------------------------

def encrypt_text(plaintext: str, key: bytes) -> str:
    """
    Chiffre un texte avec AES-256-GCM.

    Format du résultat (base64) : nonce(12B) + ciphertext + tag(16B)

    Retour
    ------
    str : données chiffrées encodées en base64.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography non disponible.")

    aesgcm    = AESGCM(key)
    nonce     = os.urandom(NONCE_SIZE)
    data      = plaintext.encode("utf-8")
    encrypted = aesgcm.encrypt(nonce, data, None)   # None = pas d'AAD

    # Concaténer nonce + ciphertext et encoder en base64
    combined = nonce + encrypted
    return base64.b64encode(combined).decode("utf-8")


def decrypt_text(encrypted_b64: str, key: bytes) -> str:
    """
    Déchiffre un texte chiffré avec AES-256-GCM.

    Lève
    ----
    cryptography.exceptions.InvalidTag si le texte a été altéré.
    """
    if not _CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography non disponible.")

    combined  = base64.b64decode(encrypted_b64.encode("utf-8"))
    nonce     = combined[:NONCE_SIZE]
    ciphertext = combined[NONCE_SIZE:]

    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(nonce, ciphertext, None)
    return plaintext.decode("utf-8")


# ---------------------------------------------------------------------------
# Chiffrement de fichier entier
# ---------------------------------------------------------------------------

def encrypt_file(input_path: str, key: bytes, output_path: str = None) -> str:
    """
    Chiffre un fichier texte et écrit le résultat chiffré.

    Retour
    ------
    str : chemin du fichier chiffré.
    """
    if output_path is None:
        output_path = input_path + ".enc"

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    encrypted = encrypt_text(content, key)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(encrypted)

    print(f"[INFO] Fichier chiffré → {output_path}")
    return output_path


def decrypt_file(encrypted_path: str, key: bytes, output_path: str = None) -> str:
    """Déchiffre un fichier précédemment chiffré par encrypt_file."""
    if output_path is None:
        output_path = encrypted_path.replace(".enc", ".dec.txt")

    with open(encrypted_path, "r", encoding="utf-8") as f:
        encrypted_b64 = f.read()

    plaintext = decrypt_text(encrypted_b64, key)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(plaintext)

    print(f"[INFO] Fichier déchiffré → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not _CRYPTO_AVAILABLE:
        print("Installez cryptography : pip install cryptography")
    else:
        print("=== Test AES-256-GCM ===")

        # Test clé aléatoire
        key = generate_key()
        message = "alice@example.com — Mot de passe: P@ssw0rd123!"

        encrypted = encrypt_text(message, key)
        decrypted = decrypt_text(encrypted, key)

        print(f"Original  : {message}")
        print(f"Chiffré   : {encrypted[:40]}...")
        print(f"Déchiffré : {decrypted}")
        print(f"Intégrité : {'✅ OK' if decrypted == message else '❌ ERREUR'}")

        # Test dérivation depuis mot de passe
        print("\n=== Test PBKDF2 ===")
        key2, salt = derive_key_from_password("MonMotDePasseSecret42!")
        enc2 = encrypt_text("Données confidentielles", key2)
        dec2 = decrypt_text(enc2, key2)
        print(f"Déchiffré (PBKDF2) : {dec2}")
