# 该文件用于实现无数据库场景下的文件上传存储与资产索引。

import json
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
FILE_DIR = UPLOAD_DIR / "files"
INDEX_PATH = UPLOAD_DIR / "index.json"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
SCENE_EXT = {".json", ".obj", ".ply", ".glb", ".gltf"}


# 该函数用于确保上传目录和索引文件存在。
def ensure_store() -> None:
    FILE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text(json.dumps({"assets": {}}, ensure_ascii=False, indent=2), encoding="utf-8")


# 该函数用于读取资产索引数据。
def load_index() -> Dict[str, Any]:
    ensure_store()
    try:
        data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("assets"), dict):
            return data
    except Exception:
        pass
    return {"assets": {}}


# 该函数用于保存资产索引数据。
def save_index(index: Dict[str, Any]) -> None:
    ensure_store()
    INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# 该函数用于根据后缀判断资产类型。
def guess_kind(ext: str) -> str:
    low = ext.lower()
    if low in IMAGE_EXT:
        return "image"
    if low in VIDEO_EXT:
        return "video"
    if low in SCENE_EXT:
        return "scene3d"
    return "other"


# 该函数用于规范化文件后缀。
def normalize_ext(filename: str, content_type: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed:
        return guessed.lower()
    return ".bin"


# 该函数用于写入上传文件并登记资产索引。
def save_asset(filename: str, content_type: str, raw: bytes) -> Dict[str, Any]:
    ensure_store()
    ext = normalize_ext(filename=filename, content_type=content_type)
    asset_id = uuid.uuid4().hex[:16]
    store_name = f"{asset_id}{ext}"
    store_path = FILE_DIR / store_name
    store_path.write_bytes(raw)

    kind = guess_kind(ext)
    meta = {
        "asset_id": asset_id,
        "origin_name": filename,
        "content_type": content_type or "",
        "ext": ext,
        "kind": kind,
        "size": len(raw),
        "path": str(store_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    index = load_index()
    index["assets"][asset_id] = meta
    save_index(index)
    return meta


# 该函数用于按资产编号读取元数据。
def get_asset(asset_id: str) -> Optional[Dict[str, Any]]:
    if not asset_id:
        return None
    index = load_index()
    data = index.get("assets", {})
    if not isinstance(data, dict):
        return None
    item = data.get(asset_id)
    if isinstance(item, dict):
        return item
    return None
