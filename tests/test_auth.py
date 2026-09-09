"""
tests/test_auth.py
===================
Tests for password hashing, registration, sign-in and watchlist isolation.

The previous auth had no password at all: the login screen listed every
registered account's name, email and phone, and one click signed you in as
any of them. These tests lock the replacement in place.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sqlite3

import pytest

import database.database as db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point the module at a throwaway database for each test."""
    path = tmp_path / "test_signals.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    db.initialise_db()
    yield path


# ── Hashing ───────────────────────────────────────────────────────────────────

def test_hash_is_salted_and_verifiable():
    a = db.hash_password("correct horse battery")
    b = db.hash_password("correct horse battery")
    assert a != b                                   # unique salt per hash
    assert "correct horse battery" not in a         # never stored in the clear
    assert db.verify_password("correct horse battery", a)
    assert db.verify_password("correct horse battery", b)


def test_verify_rejects_wrong_password():
    stored = db.hash_password("s3cret-passphrase")
    assert not db.verify_password("wrong", stored)
    assert not db.verify_password("", stored)
    assert not db.verify_password("s3cret-passphrase", None)
    assert not db.verify_password("s3cret-passphrase", "garbage")


# ── Validation ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("email,ok", [
    ("a@b.co", True),
    ("first.last@firm.co.in", True),
    ("no-at-sign", False),
    ("@nolocal.com", False),
    ("trailing@", False),
    ("has space@x.com", False),
])
def test_email_validation(email, ok):
    assert db.validate_email(email) is ok


@pytest.mark.parametrize("phone,ok", [
    ("+91 9876543210", True),
    ("9876543210", True),
    ("123", False),
    ("not-a-phone", False),
])
def test_phone_validation(phone, ok):
    assert db.validate_phone(phone) is ok


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_and_authenticate():
    ok, _, user = db.register_user("Asha Rao", "+91 9876543210", "asha@firm.com", "longenough1")
    assert ok and user is not None
    assert "password_hash" not in user               # never leaves the module

    ok, _, signed_in = db.authenticate_user("asha@firm.com", "longenough1")
    assert ok and signed_in["email"] == "asha@firm.com"


def test_register_rejects_short_password():
    ok, message, _ = db.register_user("Asha", "+91 9876543210", "a@b.com", "short")
    assert not ok and "8 characters" in message


def test_register_rejects_duplicate_email():
    db.register_user("Asha", "+91 9876543210", "dupe@firm.com", "longenough1")
    ok, message, _ = db.register_user("Other", "+91 9999999999", "dupe@firm.com", "longenough2")
    assert not ok and "already exists" in message


def test_authenticate_rejects_wrong_password():
    db.register_user("Asha", "+91 9876543210", "asha@firm.com", "longenough1")
    ok, _, user = db.authenticate_user("asha@firm.com", "wrongpassword")
    assert not ok and user is None


def test_failure_message_does_not_reveal_whether_email_exists():
    """An attacker must not be able to enumerate registered addresses."""
    db.register_user("Asha", "+91 9876543210", "known@firm.com", "longenough1")
    _, msg_known, _ = db.authenticate_user("known@firm.com", "wrongpassword")
    _, msg_unknown, _ = db.authenticate_user("nobody@firm.com", "wrongpassword")
    assert msg_known == msg_unknown


def test_no_bulk_user_listing_is_exported():
    """
    The old get_all_users() fed a pre-auth dropdown of everyone's name, email
    and phone. Only an aggregate count should be reachable now.
    """
    import database as pkg

    assert not hasattr(pkg, "get_all_users")
    assert not hasattr(pkg, "get_user_by_phone")
    assert not hasattr(pkg, "create_or_update_user")
    assert pkg.count_users() >= 0


# ── Watchlist isolation ───────────────────────────────────────────────────────

def test_watchlists_are_per_user():
    _, _, alice = db.register_user("Alice", "+91 9000000001", "alice@x.com", "longenough1")
    _, _, bob = db.register_user("Bob", "+91 9000000002", "bob@x.com", "longenough2")

    db.add_to_watchlist("RELIANCE.NS", "Reliance", user_id=alice["id"])
    db.add_to_watchlist("TCS.NS", "TCS", user_id=bob["id"])

    alice_symbols = {r["symbol"] for r in db.get_watchlist(user_id=alice["id"])}
    bob_symbols = {r["symbol"] for r in db.get_watchlist(user_id=bob["id"])}

    assert alice_symbols == {"RELIANCE.NS"}
    assert bob_symbols == {"TCS.NS"}
    assert db.is_in_watchlist("RELIANCE.NS", user_id=alice["id"])
    assert not db.is_in_watchlist("RELIANCE.NS", user_id=bob["id"])


def test_same_symbol_allowed_for_different_users():
    _, _, alice = db.register_user("Alice", "+91 9000000001", "alice@x.com", "longenough1")
    _, _, bob = db.register_user("Bob", "+91 9000000002", "bob@x.com", "longenough2")

    assert db.add_to_watchlist("INFY.NS", "Infosys", user_id=alice["id"])
    assert db.add_to_watchlist("INFY.NS", "Infosys", user_id=bob["id"])
    assert db.is_in_watchlist("INFY.NS", user_id=alice["id"])
    assert db.is_in_watchlist("INFY.NS", user_id=bob["id"])


def test_remove_only_affects_that_user():
    _, _, alice = db.register_user("Alice", "+91 9000000001", "alice@x.com", "longenough1")
    _, _, bob = db.register_user("Bob", "+91 9000000002", "bob@x.com", "longenough2")

    db.add_to_watchlist("SBIN.NS", "SBI", user_id=alice["id"])
    db.add_to_watchlist("SBIN.NS", "SBI", user_id=bob["id"])
    db.remove_from_watchlist("SBIN.NS", user_id=alice["id"])

    assert not db.is_in_watchlist("SBIN.NS", user_id=alice["id"])
    assert db.is_in_watchlist("SBIN.NS", user_id=bob["id"])
