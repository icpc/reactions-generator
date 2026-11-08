import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from reactions_generator import overlayer


def submission_payload(
    *,
    result: object = None,
    hashtag: object = "ICPC",
    time: object = 123456,
    videos: object = None,
) -> dict[str, object]:
    return {
        "team": {
            "displayName": "University",
            "customFields": {"clicsTeamFullName": "Team"},
            "hashTag": hashtag,
            "rankBefore": 12,
            "rankAfter": 2,
            "organization": {"logo": {"url": "https://example.com/logo.png"}},
        },
        "problem": {"letter": "A"},
        "time": time,
        "result": {"verdict": {"shortName": "AC", "isAccepted": True}}
        if result is None
        else result,
        "reactionVideos": [
            {"url": "https://example.com/run-webcam.mp4"},
            {"url": "https://example.com/run-desktop.mp4"},
        ]
        if videos is None
        else videos,
    }


class SubmissionTests(unittest.TestCase):
    def test_verdict_data_is_normalized_into_typed_submission(self) -> None:
        submission = overlayer.parse_submission(submission_payload())

        self.assertEqual(submission.title, "University")
        self.assertEqual(submission.subtitle, "Team")
        self.assertEqual(submission.hashtag, "#ICPC")
        self.assertEqual(submission.outcome, "AC")
        self.assertTrue(submission.success)
        self.assertEqual((submission.rank_before, submission.rank_after), (12, 2))
        self.assertEqual(submission.webcam_source, "https://example.com/run-webcam.mp4")
        self.assertEqual(
            submission.screen_source, "https://example.com/run-desktop.mp4"
        )

    def test_scoring_result_and_missing_hashtag(self) -> None:
        submission = overlayer.parse_submission(
            submission_payload(
                result={"difference": 25, "isFirstBestTeamRun": False}, hashtag=None
            )
        )

        self.assertEqual(submission.outcome, "+25")
        self.assertFalse(submission.success)
        self.assertEqual(submission.hashtag, "")

    def test_already_prefixed_hashtag_is_preserved(self) -> None:
        submission = overlayer.parse_submission(submission_payload(hashtag="#ICPC"))
        self.assertEqual(submission.hashtag, "#ICPC")

    def test_signed_video_urls_are_identified_from_the_path(self) -> None:
        webcam = "https://example.com/run-webcam.mp4?token=part.one.two"
        desktop = "https://example.com/run-desktop.mp4?token=part.one.two"
        submission = overlayer.parse_submission(
            submission_payload(videos=[{"url": webcam}, {"url": desktop}])
        )
        self.assertEqual(submission.webcam_source, webcam)
        self.assertEqual(submission.screen_source, desktop)

    def test_malformed_external_data_is_rejected(self) -> None:
        invalid_payloads: list[tuple[object, type[Exception]]] = [
            ([], TypeError),
            (submission_payload(time=True), ValueError),
            (submission_payload(hashtag=0), TypeError),
            (submission_payload(hashtag=False), TypeError),
            (
                submission_payload(
                    result={"verdict": {"shortName": "WA", "isAccepted": "false"}}
                ),
                TypeError,
            ),
            (submission_payload(videos=[]), ValueError),
        ]
        for payload, error in invalid_payloads:
            with self.subTest(payload=payload), self.assertRaises(error):
                overlayer.parse_submission(payload)

    def test_fetch_checks_status_and_normalizes_submission_ids(self) -> None:
        response = Mock()
        response.json.return_value = [{"id": 12}, {"id": "run-2"}]
        with patch.object(overlayer.requests, "get", return_value=response) as get:
            ids = overlayer.fetch_submission_ids("https://example.com")

        self.assertEqual(ids, ["12", "run-2"])
        response.raise_for_status.assert_called_once_with()
        self.assertEqual(get.call_args.kwargs["timeout"], 30)

    def test_existing_output_is_skipped_without_fetching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "12.mp4")
            output.write_bytes(b"existing render")
            with patch.object(overlayer, "fetch_submission") as fetch:
                overlayer.build_submission(
                    "https://example.com",
                    "12",
                    output_directory=directory,
                    print_progress=False,
                )
            fetch.assert_not_called()
            self.assertEqual(output.read_bytes(), b"existing render")

    def test_overwrite_preserves_existing_output_if_fetch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "12.mp4")
            output.write_bytes(b"existing render")
            with (
                patch.object(
                    overlayer,
                    "fetch_submission",
                    side_effect=ValueError("invalid submission"),
                ),
                self.assertRaises(ValueError),
            ):
                overlayer.build_submission(
                    "https://example.com",
                    "12",
                    output_directory=directory,
                    overwrite=True,
                )
            self.assertEqual(output.read_bytes(), b"existing render")


if __name__ == "__main__":
    unittest.main()
