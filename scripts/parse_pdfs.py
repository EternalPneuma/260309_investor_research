"""
MinerU PDF 解析脚本
流程：
  1. 批量申请上传 URL（POST /api/v4/file-urls/batch）
  2. PUT 上传本地 PDF 到各自的预签名 URL
  3. 轮询批量任务状态（GET /api/v4/extract-results/batch/{batch_id}）
  4. 打印结果摘要
"""

import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── 加载环境变量 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")
TOKEN = os.getenv("MinerU-API")
if not TOKEN:
    raise EnvironmentError("未找到 MinerU-API，请检查 .env 文件")

BASE_URL = "https://mineru.net/api/v4"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}
DOCUMENTS_DIR = ROOT_DIR / "documents"
POLL_INTERVAL = 10   # 秒
POLL_TIMEOUT  = 600  # 秒


def request_upload_urls(pdf_paths: list[Path]) -> tuple[str, list[str]]:
    """批量申请上传 URL。返回 (batch_id, [upload_url, ...])。"""
    payload = {
        "files": [{"name": p.name} for p in pdf_paths],
        "model_version": "vlm",
        "enable_formula": True,
        "enable_table": True,
        "language": "ch",
    }
    resp = requests.post(
        f"{BASE_URL}/file-urls/batch",
        headers=HEADERS,
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"申请上传 URL 失败: {data}")
    batch_id = data["data"]["batch_id"]
    file_urls = data["data"]["file_urls"]
    return batch_id, file_urls


def upload_file(upload_url: str, file_path: Path) -> None:
    """将本地文件 PUT 到预签名 URL（不带 Content-Type，按文档要求）。"""
    with file_path.open("rb") as f:
        resp = requests.put(upload_url, data=f, timeout=120)
    resp.raise_for_status()


def poll_batch(batch_id: str) -> list[dict]:
    """轮询批量任务，直到所有文件完成或超时。返回结果列表。"""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/extract-results/batch/{batch_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("data", {}).get("extract_result", [])

        # 统计状态
        states = {r.get("file_name", "?"): r.get("state", "") for r in results}
        done_count = sum(1 for s in states.values() if s in ("done", "failed"))
        print(f"  进度: {done_count}/{len(results)} 完成  |  " +
              "  ".join(f"{n}: {s}" for n, s in states.items()))

        if done_count == len(results):
            return results

        time.sleep(POLL_INTERVAL)

    raise TimeoutError(f"批次 {batch_id} 超时（>{POLL_TIMEOUT}s）")


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("documents/ 目录下没有找到 PDF 文件")
        return

    print(f"找到 {len(pdf_files)} 个 PDF 文件，开始处理...\n")

    # 1. 批量申请上传 URL
    print("→ 申请上传 URL...")
    batch_id, upload_urls = request_upload_urls(pdf_files)
    print(f"  batch_id: {batch_id}")

    # 2. 逐个上传
    for pdf_path, url in zip(pdf_files, upload_urls):
        print(f"→ 上传: {pdf_path.name}")
        try:
            upload_file(url, pdf_path)
            print("  上传成功")
        except Exception as e:
            print(f"  [错误] 上传失败: {e}")

    # 3. 轮询结果
    print(f"\n→ 等待解析完成（batch_id={batch_id}）...")
    results = poll_batch(batch_id)

    # 4. 打印摘要
    print(f"\n{'='*60}")
    print("解析完成，结果摘要：")
    for r in results:
        name = r.get("file_name", "?")
        state = r.get("state", "?")
        if state == "done":
            print(f"  ✓ {name}")
            print(f"    zip: {r.get('full_zip_url', 'N/A')}")
        else:
            print(f"  ✗ {name}  [{state}]  {r.get('err_msg', '')}")


if __name__ == "__main__":
    main()
