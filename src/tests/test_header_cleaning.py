"""边界测试：标题续行、正文保留，以及格式不明确时的回退。"""

from common.config import DATA_DIR, EXPERIMENT_RESULTS, FINAL_RESULTS, AUDIT_RESULTS, source_path, snapshot_common, THREAD_LIMIT
import unittest

from common.preprocessing import clean_headers


class HeaderCleaningTests(unittest.TestCase):
    def test_folded_headers_and_body(self):
        body = "From: 这行属于正文\n\n> 引用回复\n-- \n签名\n"
        text = (
            "From: author@example.com\nSubject: Re: graphics\n\tcard problem\n"
            "Organization: Example\n University\nNewsgroups: comp.graphics\n\n" + body
        )
        self.assertEqual(clean_headers(text), "Re: graphics card problem\n\n" + body)

    def test_crlf_and_whitespace_separator(self):
        text = "Subject: test\r\nLines: 2\r\n \t\r\nbody\r\n"
        self.assertEqual(clean_headers(text), "test\n\nbody\r\n")

    def test_uncertain_structure_is_unchanged(self):
        for text in (
            "Subject: no separator",
            "ordinary paragraph\n\nbody",
            "From: author\n\nbody",
            " orphan continuation\nSubject: title\n\nbody",
        ):
            with self.subTest(text=text):
                self.assertEqual(clean_headers(text), text)


if __name__ == "__main__":
    unittest.main()
