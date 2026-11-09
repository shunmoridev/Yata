from __future__ import annotations

import json
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient

from yata.api import create_app
from yata.models import SessionStore
from yata.slack import SlackSender, format_digest, format_topic
from yata.state import SessionError, SessionManager
from yata.topics import TopicPipeline

WEBHOOK = "https://hooks.slack.com/services/T000/B000/secret-part"


class Recorder:
    def __init__(self, fail_first=0):
        self.sent = []
        self.fail_first = fail_first

    @property
    def enabled(self):
        return True

    def readiness(self):
        return {"enabled": True, "reason": "fixture"}

    def deliver(self, message):
        from yata.slack import DeliveryResult
        if self.fail_first:
            self.fail_first -= 1
            return DeliveryResult("failed", "Slack が HTTP 500 を返しました。")
        self.sent.append(message["text"])
        return DeliveryResult("sent", "ok")


class SummaryLLM:
    def chat_json(self, model, *, task, system, user, schema, timeout=120.0):
        if task == "全体のまとめ":
            return {"overview": "価格は999ドルから。", "highlights": [{"text": "価格は999ドルから", "topic_ids": [user["topics"][0]["id"]]}]}
        ids = {line["text"]: line["id"] for line in user.get("lines", [])}
        price = ids.get("Pricing starts at $999.")
        return {"title": "新製品の価格", "overview": "価格は999ドルから。",
                "points": [{"text": "価格は999ドルからです。", "evidence_ids": [price]},
                           {"text": "1,099ドルの上位モデルもある。", "evidence_ids": [price]}],
                "key_facts": [{"label": "価格", "value": "$999", "evidence_id": price}]}


def session_with_summary(sender, slack=True):
    manager = SessionManager(TopicPipeline(SummaryLLM()), SessionStore(), background=False, sender=sender)
    session = manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", slack_delivery=slack)
    session.source_title = "発表会"  # as if Yata had already named the session
    manager.set_status(session, "capturing_audio")
    manager.ingest(session, [{"text": "Pricing starts at $999.", "capture_start_ms": 0, "capture_end_ms": 3000, "language": "en"}],
                   trusted_source_type="asr")
    manager.stop(session)
    return manager, session


class SenderTests(unittest.TestCase):
    def test_only_https_slack_webhooks_are_accepted_and_the_url_never_leaks(self):
        self.assertFalse(SlackSender("").enabled)
        self.assertFalse(SlackSender("http://hooks.slack.com/services/x").enabled)
        self.assertFalse(SlackSender("https://evil.example/hook").enabled)
        sender = SlackSender(WEBHOOK)
        self.assertTrue(sender.enabled)
        self.assertNotIn("secret-part", repr(sender))
        self.assertNotIn("secret-part", json.dumps(sender.readiness(), ensure_ascii=False))

    def test_delivery_posts_json_and_reports_failures_without_the_url(self):
        calls = []

        class Response:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): return False
            def read(self, _limit): return b"ok"

        def opener(request, timeout):
            calls.append(json.loads(request.data.decode()))
            return Response()

        result = SlackSender(WEBHOOK, opener=opener).deliver({"text": "要約"})
        self.assertEqual((result.state, calls), ("sent", [{"text": "要約"}]))

        def failing(request, timeout):
            raise urllib.error.HTTPError(WEBHOOK, 404, "no_service", {}, None)

        failed = SlackSender(WEBHOOK, opener=failing).deliver({"text": "要約"})
        self.assertEqual(failed.state, "failed")
        self.assertIn("404", failed.reason)
        self.assertNotIn("secret-part", failed.reason)
        self.assertEqual(SlackSender("").deliver({"text": "x"}).state, "local_only")

    def test_messages_hold_only_verified_text(self):
        topic = {"index": 2, "title": "t", "start_ms": 61_000, "end_ms": 125_000, "summary": {
            "title": "新製品", "overview": "概要。", "key_facts": [{"label": "価格", "value": "$999", "evidence_id": "e1"}],
            "points": [{"text": "確認済み", "needs_review": False, "evidence_ids": ["e1"]},
                       {"text": "", "needs_review": True, "evidence_ids": ["e2"]}]}}
        text = format_topic("発表会", topic)
        self.assertIn("*発表会 · 話題 2: 新製品*  (00:01:01–00:02:05)", text)
        self.assertIn("• 確認済み", text)
        self.assertEqual(text.count("• "), 1)
        self.assertIn("価格: $999", text)
        digest = format_digest("", {"overview": "全体", "highlights": [{"text": "a", "needs_review": False}, {"text": "", "needs_review": True}]})
        self.assertEqual(digest.count("• "), 1)


class ManagerDeliveryTests(unittest.TestCase):
    def test_opted_in_session_sends_each_verified_topic_and_the_digest_once(self):
        sender = Recorder()
        manager, session = session_with_summary(sender)
        self.assertEqual(session.topics[0].delivery_state, "sent")
        self.assertEqual(session.digest_delivery_state, "sent")
        self.assertEqual(len(sender.sent), 2)
        self.assertIn("価格は999ドルからです。", sender.sent[0])
        self.assertNotIn("1,099", sender.sent[0], "a point held for review is not sent")
        self.assertNotIn("Pricing starts", sender.sent[0], "transcript lines are not sent")
        manager.deliver(session)
        self.assertEqual(len(sender.sent), 2)

    def test_sessions_without_opt_in_send_nothing(self):
        sender = Recorder()
        _manager, session = session_with_summary(sender, slack=False)
        self.assertEqual(sender.sent, [])
        self.assertEqual(session.topics[0].delivery_state, "local_only")

    def test_failure_is_shown_and_retried_only_on_request(self):
        sender = Recorder(fail_first=1)
        manager, session = session_with_summary(sender)
        topic = session.topics[0]
        self.assertEqual((topic.delivery_state, topic.delivery_reason), ("failed", "Slack が HTTP 500 を返しました。"))
        manager.deliver(session)
        self.assertEqual(topic.delivery_state, "failed", "no silent automatic retry")
        manager.redeliver(session, topic.id)
        self.assertEqual(topic.delivery_state, "sent")
        with self.assertRaises(SessionError):
            manager.redeliver(session, topic.id)

    def test_opt_in_requires_a_configured_webhook(self):
        manager = SessionManager(TopicPipeline(SummaryLLM()), SessionStore(), background=False)
        with self.assertRaises(SessionError):
            manager.create(source_url="", source_title="", source_kind="share", selected_model="qwen3:8b", slack_delivery=True)


class LocalWebhookTests(unittest.TestCase):
    def test_real_http_post_reaches_a_local_mock_webhook(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *_args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sender = SlackSender(f"http://127.0.0.1:{server.server_port}/hook", allowed_hosts=frozenset({"127.0.0.1"}), require_https=False)
            _manager, session = session_with_summary(sender)
        finally:
            server.shutdown()
        self.assertEqual([item["text"].splitlines()[0] for item in received],
                         ["*発表会 · 話題 1: 新製品の価格*  (00:00:00–00:00:03)", "*発表会 · 全体のまとめ*"])
        self.assertEqual(session.topics[0].delivery_state, "sent")


class ApiTests(unittest.TestCase):
    def test_health_reports_slack_and_the_redeliver_route_checks_state(self):
        class Ollama:
            def models(self): return ["qwen3:8b"]
            def allows(self, model): return model == "qwen3:8b"
            def chat_json(self, *_a, **_k): return {}

        app = create_app(access_token="t", ollama=Ollama(), asr=None, slack=SlackSender(""))
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            client.get("/")
            health = client.get("/api/v1/health").json()
            self.assertFalse(health["slack"]["enabled"])
            refused = client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b", "slack_delivery": True})
            self.assertEqual(refused.status_code, 422)
            session_id = client.post("/api/v1/sessions", json={"source_kind": "share", "selected_model": "qwen3:8b"}).json()["session"]["id"]
            self.assertEqual(client.post(f"/api/v1/sessions/{session_id}/topics/missing/deliver").status_code, 409)


if __name__ == "__main__":
    unittest.main()
