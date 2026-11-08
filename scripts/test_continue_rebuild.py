"""Cooldown scheduling must preserve the configured default model."""

import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('continue_rebuild', Path(__file__).with_name('continue-rebuild.py'))
RESUME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESUME)


class CooldownTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 3, 10, 0, tzinfo=timezone.utc)

    def test_longest_reported_provider_cooldown_is_honored(self):
        output = 'Cooldown wait 7100s.\nCooldown wait 7145s.'
        self.assertEqual(RESUME.cooldown_until(output, self.now), self.now + timedelta(seconds=7145))

    def test_other_provider_errors_do_not_invent_a_cooldown(self):
        self.assertIsNone(RESUME.cooldown_until('HTTP 503 network unavailable', self.now))

    def test_image_quota_exhaustion_is_retried_without_failing_text_work(self):
        self.assertTrue(RESUME.transient_image_failure(
            '{"error":{"code":"image_capacity_exhausted","message":"No eligible accounts"}}'))

    def test_provider_concurrency_limits_are_temporary(self):
        self.assertTrue(RESUME.transient_image_failure('Image provider HTTP 429: concurrency exceeded'))

    def test_invalid_image_requests_are_not_hidden_as_quota_errors(self):
        self.assertFalse(RESUME.transient_image_failure('{"error":{"code":"invalid_request"}}'))

    def test_local_text_service_connection_failure_waits_for_recovery(self):
        self.assertTrue(RESUME.transient_text_failure('All connection attempts failed'))
        self.assertTrue(RESUME.transient_text_failure('Text model HTTP 503 unavailable'))

    def test_text_validation_failure_is_not_an_infrastructure_retry(self):
        self.assertFalse(RESUME.transient_text_failure('Draft validation failed: missing illustration anchor'))


if __name__ == '__main__':
    unittest.main()
