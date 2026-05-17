# 该文件用于提供第二阶段前端服务：上传、检测、深度、场景图、导出五步 API。
import argparse
import cgi
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

from asset_store import get_asset, save_asset
from config import get_config_value, load_config
from demo_core import SpatialDemoCore


ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"


# 该函数用于解析前端服务启动参数。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial-VLA Stage2 Frontend Server")
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
        self.detector_model = str(get_config_value(conf, "models", "detector_model", "IDEA-Research/grounding-dino-tiny"))
        self.depth_model = str(get_config_value(conf, "models", "depth_model", "depth-anything/Depth-Anything-V2-Small-hf"))
        self.box_thr = float(get_config_value(conf, "models", "detector_box_threshold", 0.25))
        self.text_thr = float(get_config_value(conf, "models", "detector_text_threshold", 0.25))

        # 该变量用于复用同一个核心实例，避免每次请求重复加载模型。
        self._core = SpatialDemoCore(
            detector_model=self.detector_model,
            depth_model=self.depth_model,
            box_thr=self.box_thr,
            text_thr=self.text_thr,
        )
        # 该锁用于串行化推理，减少并发推理造成的卡顿。
        self._infer_lock = threading.Lock()

    # 该函数用于获取核心执行器实例。
    def core(self) -> SpatialDemoCore:
        return self._core

    # 该函数用于根据 asset_id 解析上传图片路径。
    def resolve_image(self, payload: Dict[str, Any]) -> str:
        asset_id = str(payload.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError("missing asset_id")
        meta = get_asset(asset_id)
        if meta is None:
            raise ValueError(f"asset_id not found: {asset_id}")
        if str(meta.get("kind", "")) != "image":
            raise ValueError(f"only image is supported, got: {meta.get('kind', '')}")
        path = str(meta.get("path", "")).strip()
        if not path:
            raise ValueError(f"asset path missing for: {asset_id}")
        return path

    # 该函数用于执行第一步物体检测。
    def detect(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.resolve_image(payload)
        with self._infer_lock:
            return self.core().detect_objects(image_path=image_path)

    # 该函数用于执行第二步深度估计。
    def depth(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.resolve_image(payload)
        objects = payload.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("depth requires objects list")
        with self._infer_lock:
            return self.core().estimate_depth(image_path=image_path, objects=objects)

    # 该函数用于执行第三步 Scene Graph 构建。
    def graph(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        objects = payload.get("objects")
        depth_by_object = payload.get("depth_by_object")
        image_size = payload.get("image_size")
        if not isinstance(objects, list) or not objects:
            raise ValueError("graph requires objects list")
        if not isinstance(depth_by_object, dict):
            raise ValueError("graph requires depth_by_object object")
        if not isinstance(image_size, dict):
            raise ValueError("graph requires image_size object")
        scene_graph = self.core().build_scene_graph(
            objects=objects,
            depth_by_object=depth_by_object,
            image_size=image_size,
        )
        return {"scene_graph": scene_graph}

    # 该函数用于执行第五步 JSON 导出响应。
    def export(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scene_graph = payload.get("scene_graph")
        if not isinstance(scene_graph, dict):
            raise ValueError("export requires scene_graph")
        return {"file_name": "scene_graph.json", "content": scene_graph}


class FrontHandler(BaseHTTPRequestHandler):
    service: FrontService | None = None

    # 该函数用于统一返回 JSON 响应。
    def send_json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # 该函数用于统一返回文本响应。
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
            raise ValueError("request body must be a JSON object")
        return data

    # 该函数用于解析 multipart/form-data 上传文件。
    def read_upload(self) -> Tuple[str, str, bytes]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("upload endpoint only accepts multipart/form-data")
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
            raise ValueError("file field is missing")
        if isinstance(item, list):
            item = item[0]
        name = str(getattr(item, "filename", "") or "").strip()
        if not name:
            raise ValueError("uploaded file name is empty")
        raw = item.file.read() if getattr(item, "file", None) else b""
        if not raw:
            raise ValueError("uploaded file is empty")
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
            if path == "/api/detect":
                self.send_json(200, {"ok": True, "result": self.service.detect(payload)})
                return
            if path == "/api/depth":
                self.send_json(200, {"ok": True, "result": self.service.depth(payload)})
                return
            if path == "/api/graph":
                self.send_json(200, {"ok": True, "result": self.service.graph(payload)})
                return
            if path == "/api/export":
                self.send_json(200, {"ok": True, "result": self.service.export(payload)})
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
