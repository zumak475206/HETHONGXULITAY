"""
tools/sync_model.py
===================
Công cụ đồng bộ tự động mô hình ONNX mới nhất từ Google Drive hoặc liên kết mạng về thư mục weights/ của dự án.

Chức năng:
1. Tải file trọng số (model.onnx, model.pt) từ Google Drive (hỗ trợ cả gdown và urllib fallback).
2. Kiểm tra tính toàn vẹn bằng mã băm MD5 để không tải lại nếu mô hình local đã là bản mới nhất.
3. Ghi file an toàn (Atomic write via .tmp) chống lỗi file hỏng khi mất mạng giữa chừng.
4. Bảo vệ ổ C: Mặc định lưu trữ trên ổ D (weights/ trong thư mục dự án).

Cách dùng:
    python tools/sync_model.py --file_id <GOOGLE_DRIVE_FILE_ID>
    python tools/sync_model.py --url <SHARING_URL> --output weights/gesture_bilstm_multiscale.onnx
"""

import os
import sys
import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

# Đảm bảo import config từ dự án
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config


def calculate_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Tính toán mã băm MD5 của một file."""
    if not file_path.exists():
        return ""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def extract_file_id(url_or_id: str) -> str:
    """Trích xuất Google Drive File ID từ URL chia sẻ hoặc ID thô."""
    if not url_or_id:
        return ""
    url_or_id = url_or_id.strip()
    if "/" in url_or_id:
        # Ví dụ: https://drive.google.com/file/d/1a2b3c4d.../view?usp=sharing
        if "/d/" in url_or_id:
            parts = url_or_id.split("/d/")
            if len(parts) > 1:
                return parts[1].split("/")[0].split("?")[0]
        elif "id=" in url_or_id:
            parts = url_or_id.split("id=")
            if len(parts) > 1:
                return parts[1].split("&")[0]
    return url_or_id


def _matches_md5(actual_md5: str, expected_md5: Optional[str]) -> bool:
    """Kiem tra ma MD5 khop chinh xac hoac khop voi prefix (short hash)."""
    if not expected_md5:
        return False
    act = actual_md5.lower().strip()
    exp = expected_md5.lower().strip()
    if len(exp) >= 32:
        return act == exp
    return act.startswith(exp)


def _is_binary_model(file_path: Path) -> bool:
    """Kiem tra xem file tai ve co phai la binary hop le hay bi tra ve trang HTML Google Drive."""
    if not file_path.exists() or file_path.stat().st_size < 512:
        return False
    try:
        with open(file_path, "rb") as f:
            header = f.read(512).lower()
        if b"<!doctype" in header or b"<html" in header or b"<head" in header or b"google.com" in header:
            return False
        return True
    except Exception:
        return False


def download_model(
    source: str,
    output_path: Path,
    expected_md5: Optional[str] = None,
    force: bool = False,
) -> bool:
    """
    Tải file mô hình về output_path.
    Nếu file đã tồn tại và mã hash MD5 trùng khớp thì bỏ qua (Zero-waste).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_id = extract_file_id(source)

    if output_path.exists() and not force:
        local_md5 = calculate_md5(output_path)
        if expected_md5 and _matches_md5(local_md5, expected_md5):
            print(f"[*] Mo hinh tai {output_path.name} da la ban moi nhat (MD5: {local_md5[:8]}...). Bo qua tai.")
            return True

    print(f"[*] Bat dau tai mo hinh tu: {source}")
    print(f"[*] Dich luu: {output_path}")

    temp_file = output_path.with_suffix(output_path.suffix + ".tmp")

    # Chien luoc 1: Thu dung gdown neu da cai dat
    download_success = False
    try:
        import gdown
        print("[*] Dang tai qua gdown...")
        if "http://" in source or "https://" in source:
            gdown.download(source, str(temp_file), quiet=False, fuzzy=True)
        else:
            gdown.download(id=file_id, output=str(temp_file), quiet=False)
        if temp_file.exists() and _is_binary_model(temp_file):
            download_success = True
    except ImportError:
        pass
    except Exception as e:
        print(f"[!] gdown canh bao: {e}")

    # Chien luoc 2: Fallback qua urllib truc tiep tu Google Drive kem xu ly cookie/confirm
    if not download_success:
        print("[*] Dang tai qua urllib fallback...")
        if file_id and ("/" not in file_id):
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        else:
            direct_url = source

        import http.cookiejar
        cookie_jar = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

        req = urllib.request.Request(
            direct_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        try:
            with opener.open(req, timeout=60) as resp, open(temp_file, "wb") as out_f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)

            # Kiem tra neu Google Drive tra ve trang xac nhan virus (download confirmation page)
            if temp_file.exists() and not _is_binary_model(temp_file):
                with open(temp_file, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                import re
                confirm_match = re.search(r"confirm=([0-9A-Za-z_]+)", html_content)
                if confirm_match:
                    confirm_code = confirm_match.group(1)
                    confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_code}&id={file_id}"
                    print(f"[*] Phat hien xac nhan Google Drive: confirm={confirm_code}. Dang tai tiep...")
                    req_confirm = urllib.request.Request(
                        confirm_url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                    )
                    with opener.open(req_confirm, timeout=60) as resp2, open(temp_file, "wb") as out_f2:
                        while True:
                            chunk = resp2.read(65536)
                            if not chunk:
                                break
                            out_f2.write(chunk)

            if temp_file.exists() and _is_binary_model(temp_file):
                download_success = True
            elif temp_file.exists():
                print(f"[!] Canh bao: Noi dung tai ve khong phai la file nhi phan hop le (co the la HTML loi).")
        except Exception as e:
            print(f"[!] Loi tai urllib: {e}")

    if not download_success:
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        print(f"[!] That bai: Khong the tai mo hinh hop le tu nguon da cung cap.")
        return False

    # Kiem tra ma bam MD5 neu co
    downloaded_md5 = calculate_md5(temp_file)
    if expected_md5 and not _matches_md5(downloaded_md5, expected_md5):
        print(f"[!] Sai lech checksum MD5: Nhan {downloaded_md5}, mong doi {expected_md5}.")
        temp_file.unlink(missing_ok=True)
        return False

    # Ghi de file chinh thuc an toan (Atomic rename)
    temp_file.replace(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[+] Tai mo hinh thanh cong: {output_path.name} ({file_size_mb:.2f} MB | MD5: {downloaded_md5[:8]}...)")
    return True


def main():
    parser = argparse.ArgumentParser(description="HETHONGXULITAY - Dong bo mo hinh tu Google Drive")
    parser.add_argument("--file_id", type=str, default="", help="Google Drive File ID")
    parser.add_argument("--url", type=str, default="", help="Google Drive sharing URL hoac HTTP direct link")
    parser.add_argument("--output", type=str, default=str(config.MODEL_ONNX_PATH), help="Duong dan file luu local")
    parser.add_argument("--md5", type=str, default=None, help="Ma hash MD5 mong doi de kiem tra toan ven")
    parser.add_argument("--force", action="store_true", help="Bat buoc tai lai du file da ton tai")
    args = parser.parse_args()

    source = args.url if args.url else args.file_id
    if not source:
        print("[!] Loi: Ban can chi dinh it nhat mot trong hai tham so: --file_id hoac --url.")
        sys.exit(1)

    ok = download_model(source=source, output_path=Path(args.output), expected_md5=args.md5, force=args.force)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
