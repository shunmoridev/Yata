from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from yata.api import create_app
from yata.features import Features, env_name, load_features
from yata.models import SessionStore
from yata.ollama import SUMMARY_MODEL_ALLOWLIST
from yata.slack import SlackSender
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline

WEBHOOK = "https://hooks.slack.com/services/T000/B000/secret-part"


class Ollama:
    def models(self):
        return ["qwen3:8b"]

    def allows(self, model):
        return model in SUMMARY_MODEL_ALLOWLIST

    def chat_json(self, *_args, **_kwargs):
        return {}


def client(features, slack=None):
    app = create_app(access_token="t", ollama=Ollama(), asr=None, slack=slack, features=features)
    test_client = TestClient(app, base_url="http://127.0.0.1:8765")
    return test_client, test_client.get("/")


class FeatureTests(unittest.TestCase):
    def test_every_toggle_is_off_unless_its_variable_says_on(self):
        self.assertEqual(load_features({}), Features(theme_switch=False, slack=False))
        self.assertEqual(env_name("theme_switch"), "YATA_FEATURE_THEME_SWITCH")
        for value in ("on", "1", "TRUE", " yes "):
            self.assertTrue(load_features({"YATA_FEATURE_SLACK": value}).slack, value)
        for value in ("off", "0", "", "enabled"):
            self.assertFalse(load_features({"YATA_FEATURE_SLACK": value}).slack, value)
        both = load_features({"YATA_FEATURE_SLACK": "on", "YATA_FEATURE_THEME_SWITCH": "on"})
        self.assertEqual(both.enabled(), ["theme_switch", "slack"])
        self.assertEqual(both.public(), {"theme_switch": True, "slack": True})

    def test_the_page_and_health_say_which_features_are_on(self):
        off, page = client(Features())
        self.assertIn('<html lang="ja" data-theme="soft" data-features="" data-locale="">', page.text)
        self.assertEqual(off.get("/api/v1/health").json()["features"], {"theme_switch": False, "slack": False})
        on, page = client(Features(theme_switch=True, slack=True))
        self.assertIn('data-features="theme_switch slack"', page.text)
        self.assertEqual(on.get("/api/v1/health").json()["features"], {"theme_switch": True, "slack": True})

    def test_slack_off_refuses_delivery_even_with_a_webhook_configured(self):
        test_client, _ = client(Features(slack=False), slack=SlackSender(WEBHOOK))
        health = test_client.get("/api/v1/health").json()["slack"]
        self.assertFalse(health["enabled"])
        self.assertIn("YATA_FEATURE_SLACK", health["reason"])
        self.assertNotIn("hooks.slack.com", str(health))
        body = {"source_kind": "share", "selected_model": "qwen3:8b"}
        self.assertEqual(test_client.post("/api/v1/sessions", json={**body, "slack_delivery": True}).status_code, 422)
        session_id = test_client.post("/api/v1/sessions", json=body).json()["session"]["id"]
        refused = test_client.post(f"/api/v1/sessions/{session_id}/topics/any/deliver")
        self.assertEqual(refused.status_code, 409)
        self.assertIn("機能トグル", refused.json()["detail"])

    def test_slack_on_uses_the_configured_webhook(self):
        test_client, _ = client(Features(slack=True), slack=SlackSender(WEBHOOK))
        self.assertTrue(test_client.get("/api/v1/health").json()["slack"]["enabled"])
        created = test_client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b", "slack_delivery": True})
        self.assertEqual(created.status_code, 200, created.text)
        self.assertTrue(created.json()["session"]["slack_delivery"])

    def test_a_switched_off_sender_never_opens_a_connection(self):
        opened = []
        sender = SlackSender(WEBHOOK, switched_off=True, opener=lambda *args, **kwargs: opened.append(args))
        self.assertFalse(sender.enabled)
        self.assertEqual(sender.deliver({"text": "x"}).state, "local_only")
        self.assertEqual(opened, [])
        manager = SessionManager(TopicPipeline(Ollama()), SessionStore(), background=False, sender=sender)
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", slack_delivery=True)


if __name__ == "__main__":
    unittest.main()
