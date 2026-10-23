"""
tests/test_sync_model.py
========================
Unit tests kiem thu cong cu dong bo mo hinh Google Drive (tools/sync_model.py).
"""

import unittest
import tempfile
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.sync_model import (
    calculate_md5,
    extract_file_id,
    _matches_md5,
    _is_binary_model,
    download_model
)


class TestSyncModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=str(ROOT / "dataset"))
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_calculate_md5(self):
        """Kiem tra tinh toan ma bam MD5 khop voi hashlib chuan."""
        test_file = self.test_dir / "sample.bin"
        sample_data = b"HETHONGXULITAY_MODEL_WEIGHTS_MOCK_158D" * 20
        test_file.write_bytes(sample_data)

        expected = hashlib.md5(sample_data).hexdigest()
        actual = calculate_md5(test_file)
        self.assertEqual(actual, expected)

    def test_extract_file_id(self):
        """Kiem tra trich xuat Google Drive File ID tu cac dinh dang URL khac nhau."""
        raw_id = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ_01234"
        url_d = f"https://drive.google.com/file/d/{raw_id}/view?usp=sharing"
        url_query = f"https://drive.google.com/open?id={raw_id}&authuser=0"
        url_uc = f"https://drive.google.com/uc?export=download&id={raw_id}"

        self.assertEqual(extract_file_id(raw_id), raw_id)
        self.assertEqual(extract_file_id(url_d), raw_id)
        self.assertEqual(extract_file_id(url_query), raw_id)
        self.assertEqual(extract_file_id(url_uc), raw_id)
        self.assertEqual(extract_file_id(""), "")

    def test_matches_md5(self):
        """Kiem tra co che so khop MD5: ho tro ca full 32-char va short prefix."""
        full_hash = "fa0aef27170283928701874c41eed398"
        self.assertTrue(_matches_md5(full_hash, "fa0aef27170283928701874c41eed398"))
        self.assertTrue(_matches_md5(full_hash, "FA0AEF27170283928701874C41EED398"))
        # Short prefix (8-char)
        self.assertTrue(_matches_md5(full_hash, "fa0aef27"))
        # Short prefix (16-char)
        self.assertTrue(_matches_md5(full_hash, "fa0aef2717028392"))
        # Mismatch
        self.assertFalse(_matches_md5(full_hash, "bbbb1234"))
        self.assertFalse(_matches_md5(full_hash, None))
        self.assertFalse(_matches_md5(full_hash, ""))

    def test_is_binary_model_rejects_html(self):
        """Kiem tra _is_binary_model loai bo file HTML va file qua nho."""
        # File HTML gia lap tu Google Drive Warning / Error 404
        html_file = self.test_dir / "error.html"
        html_file.write_text("<!DOCTYPE html><html><head><title>Google Drive - Warning</title></head><body>Confirm download</body></html>", encoding="utf-8")
        self.assertFalse(_is_binary_model(html_file))

        # File qua nho (< 512 bytes)
        small_file = self.test_dir / "tiny.bin"
        small_file.write_bytes(b"0" * 100)
        self.assertFalse(_is_binary_model(small_file))

        # File binary hop le (> 512 bytes va khong chua tag HTML)
        valid_binary = self.test_dir / "model.onnx"
        valid_binary.write_bytes(b"\x08\x01\x12\x04ONNX" + b"\x00" * 1000)
        self.assertTrue(_is_binary_model(valid_binary))

    def test_download_model_skip_existing_matching_md5(self):
        """Kiem tra download_model khong tai lai khi file da co san va khop MD5."""
        model_file = self.test_dir / "gesture_model.onnx"
        data = b"\x08\x01\x12\x04ONNX" + b"\x00" * 2048
        model_file.write_bytes(data)

        md5_val = hashlib.md5(data).hexdigest()
        # Test voi prefix 8 ky tu
        res_prefix = download_model(source="dummy_source", output_path=model_file, expected_md5=md5_val[:8])
        self.assertTrue(res_prefix)

        # Test voi full hash
        res_full = download_model(source="dummy_source", output_path=model_file, expected_md5=md5_val)
        self.assertTrue(res_full)


if __name__ == "__main__":
    unittest.main()
