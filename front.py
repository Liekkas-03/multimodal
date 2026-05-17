# 该文件用于提供第一阶段最小可运行 Demo 的前端服务与最小 API 路由。
import argparse
import cgi
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from asset_store import get_asset, save_asset
from config import get_config_value, load_config
from demo_core import SpatialDemoCore


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"


# 该函数用于解析前端服务启动参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial-VLA Minimal Frontend Server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--config", type=str, default="config.toml")
    return parser.parse_args()


# 该函数用于整理资产元信息，避免返回无关字段。
def asset_public(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset_id": meta.get("asset_id", ""),
        "origin_name": meta.get("origin_name", ""),
        "kind": meta.get("kind", ""),
        "ext": meta.get("ext", ""),
        "size": meta.get("size", 0),
        "created_at": meta.get("created_at", ""),
        "path": meta.get("path", ""),
    }


class FrontService:
    # 该函数用于初始化服务并读取模型配置。
    def __init__(self, config_path: str) -> None:
        conf = load_config(config_path)
        self.api_base = str(get_config_value(conf, "vision", "api_base", ""))
        self.api_key = str(get_config_value(conf, "vision", "api_key", ""))
        self.model = str(get_config_value(conf, "vision", "model", ""))
        self.timeout = int(get_config_value(conf, "vision", "timeout", 90))

    # 该函数用于构建核心模型调用器。
    def core(self) -> SpatialDemoCore:
        return SpatialDemoCore(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            timeout=self.timeout,
        )

    # 该函数用于根据 asset_id 解析上传图片路径。
    def resolve_image(self, payload: Dict[str, Any]) -> str:
        asset_id = str(payload.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("缺少 asset_id。")
        meta = get_asset(asset_id)
        if meta is None:
            raise ValueError(f"asset_id 不存在: {asset_id}")
        if str(meta.get("kind", "")) != "image":
            raise ValueError(f"当前仅支持 image，收到: {meta.get('kind', '')}")
        path = str(meta.get("path", "")).strip()
        if not path:
            raise ValueError(f"asset_id 缺少路径: {asset_id}")
        return path

    # 该函数用于执行场景理解。
    def analyze(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.resolve_image(payload)
        analysis = self.core().analyze_scene(image_path=image_path)
        return {"analysis": analysis}

    # 该函数用于执行动作计划生成。
    def plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.resolve_image(payload)
        task = str(payload.get("task", "")).strip()
        if not task:
            raise ValueError("任务不能为空。")
        analysis = payload.get("analysis")
        if not isinstance(analysis, dict):
            raise ValueError("plan 需要 analysis。")
        action_plan = self.core().generate_action_plan(image_path=image_path, task=task, context=analysis)
        return {"action_plan": action_plan}


class FrontHandler(BaseHTTPRequestHandler):
    service: FrontService | None = None

    # 该函数用于统一发送 JSON 响应。
    def send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # 该函数用于统一发送文本响应。
    def send_text(self, code: int, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # 该函数用于返回静态文件内容。
    def send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_text(404, "Not Found")
            return
        mime, _ = mimetypes.guess_type(str(path))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", f"{mime or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # 该函数用于读取并解析 JSON 请求体。
    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象。")
        return data

    # 该函数用于解析 multipart/form-data 上传文件。
    def read_upload(self) -> tuple[str, str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("上传接口只接受 multipart/form-data。")
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )
        item = form["file"] if "file" in form else None
        if item is None:
            raise ValueError("未检测到 file 字段。")
        if isinstance(item, list):
            item = item[0]
        name = str(getattr(item, "filename", "") or "").strip()
        if not name:
            raise ValueError("上传文件缺少文件名。")
        raw = item.file.read() if getattr(item, "file", None) else b""
        if not raw:
            raise ValueError("上传文件为空。")
        ctype = str(getattr(item, "type", "") or "")
        return name, ctype, raw

    # 该函数用于处理 GET 请求路由。
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_file(UI_DIR / "index.html")
            return
        self.send_text(404, "Not Found")

    # 该函数用于处理 POST 请求路由。
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if self.service is None:
            self.send_json(500, {"ok": False, "error": "service not ready"})
            return
        try:
            if path == "/api/upload":
                name, ctype, raw = self.read_upload()
                meta = save_asset(filename=name, content_type=ctype, raw=raw)
                self.send_json(200, {"ok": True, "asset": asset_public(meta)})
                return
            payload = self.read_json()
            if path == "/api/analyze":
                self.send_json(200, {"ok": True, "result": self.service.analyze(payload)})
                return
            if path == "/api/plan":
                self.send_json(200, {"ok": True, "result": self.service.plan(payload)})
                return
            self.send_text(404, "Not Found")
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


# 该函数用于启动前端服务。
def main() -> None:
    args = parse_args()
    service = FrontService(config_path=args.config)
    FrontHandler.service = service
    UI_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), FrontHandler)
    print(f"Frontend running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
