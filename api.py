# 该文件用于提供第一阶段最小可运行 Demo 的后端 API（默认挂在 /v1）。
import argparse
import cgi
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from asset_store import save_asset
from front import FrontService, asset_public


# 该函数用于解析后端服务启动参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial-VLA Minimal Backend API")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=str, default="config.toml")
    return parser.parse_args()


class ApiHandler(BaseHTTPRequestHandler):
    service: FrontService | None = None

    # 该函数用于统一发送 JSON 响应。
    def send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
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

    # 该函数用于处理 GET 路由。
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/v1", "/v1/"}:
            self.send_json(200, {"ok": True, "routes": ["/v1/upload", "/v1/analyze", "/v1/plan"]})
            return
        self.send_json(404, {"ok": False, "error": "not found"})

    # 该函数用于处理 POST 路由。
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if self.service is None:
            self.send_json(500, {"ok": False, "error": "service not ready"})
            return
        try:
            if path == "/v1/upload":
                name, ctype, raw = self.read_upload()
                meta = save_asset(filename=name, content_type=ctype, raw=raw)
                self.send_json(200, {"ok": True, "asset": asset_public(meta)})
                return
            payload = self.read_json()
            if path == "/v1/analyze":
                self.send_json(200, {"ok": True, "result": self.service.analyze(payload)})
                return
            if path == "/v1/plan":
                self.send_json(200, {"ok": True, "result": self.service.plan(payload)})
                return
            self.send_json(404, {"ok": False, "error": "not found"})
        except Exception as exc:
            self.send_json(500, {"ok": False, "error": str(exc)})


# 该函数用于启动后端 API 服务。
def main() -> None:
    args = parse_args()
    service = FrontService(config_path=args.config)
    ApiHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"Backend running at http://{args.host}:{args.port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
