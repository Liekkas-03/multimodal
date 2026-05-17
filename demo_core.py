# 该文件用于实现第二阶段核心能力：开放词表检测、深度估计、空间关系推断与 Scene Graph 生成。
import base64
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


CLASSES = [
    "sofa",
    "coffee table",
    "dining table",
    "chair",
    "laptop",
    "mug",
    "plant",
    "tv",
    "shelf",
    "book",
    "bottle",
    "remote control",
    "cabinet",
    "floor",
]

NAME_MAP = {
    "sofa": "sofa",
    "coffee table": "coffee_table",
    "dining table": "dining_table",
    "chair": "chair",
    "laptop": "laptop",
    "mug": "mug",
    "plant": "plant",
    "tv": "tv",
    "shelf": "shelf",
    "book": "book",
    "bottle": "bottle",
    "remote control": "remote_control",
    "cabinet": "cabinet",
    "floor": "floor",
}

MOVABLE = {"mug", "bottle", "book", "remote_control"}
GRASPABLE = {"mug", "bottle", "remote_control"}
PLACEABLE = {"coffee_table", "dining_table", "shelf", "cabinet"}
FRAGILE = {"laptop", "tv"}
LIQUID_RISK = {"mug", "bottle"}
CONTAINER = {"mug", "bottle"}


@dataclass
class Obj:
    id: str
    name: str
    bbox: List[int]
    confidence: float
    avg_depth: float = 0.0


# 该函数用于把图像编码为 base64 PNG 文本。
def image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# 该函数用于读取上传图像并统一转为 RGB。
def load_rgb(path: str) -> Image.Image:
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"图片不存在: {path}")
    return Image.open(file).convert("RGB")


# 该函数用于把标签标准化为固定类别名。
def normalize_label(label: str) -> str:
    raw = str(label).strip().lower().replace("-", " ")
    raw = " ".join(raw.split())
    if raw in NAME_MAP:
        return NAME_MAP[raw]
    for key, value in NAME_MAP.items():
        if key in raw:
            return value
    return raw.replace(" ", "_")


# 该函数用于计算两个 bbox 的 IoU。
def iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# 该函数用于将检测结果按类别做简单 NMS 去重。
def class_nms(items: List[Dict[str, Any]], thr: float = 0.55) -> List[Dict[str, Any]]:
    keep: List[Dict[str, Any]] = []
    for item in sorted(items, key=lambda x: float(x["score"]), reverse=True):
        ok = True
        for saved in keep:
            if item["name"] == saved["name"] and iou(item["bbox"], saved["bbox"]) > thr:
                ok = False
                break
        if ok:
            keep.append(item)
    return keep


# 该函数用于绘制 bbox 叠加图。
def draw_boxes(image: Image.Image, objs: List[Obj]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for obj in objs:
        x1, y1, x2, y2 = obj.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(40, 255, 120), width=3)
        title = f"{obj.id}:{obj.name} {obj.confidence:.2f}"
        tx = max(2, x1)
        ty = max(2, y1 - 16)
        draw.rectangle([tx, ty, tx + min(320, len(title) * 8 + 8), ty + 14], fill=(0, 0, 0))
        draw.text((tx + 3, ty + 1), title, fill=(255, 255, 255))
    return canvas


# 该函数用于把浮点深度图标准化到 0~1。
def normalize_map(depth: np.ndarray) -> np.ndarray:
    arr = depth.astype(np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


# 该函数用于约束 bbox 在图像范围内。
def clamp_box(box: List[int], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]


# 该函数用于计算 bbox 中心点。
def box_center(box: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


# 该函数用于计算两个 bbox 的边界距离（像素）。
def box_gap(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(0, max(bx1 - ax2, ax1 - bx2))
    dy = max(0, max(by1 - ay2, ay1 - by2))
    return float(math.sqrt(dx * dx + dy * dy))


# 该函数用于根据类别补充对象属性。
def obj_attrs(name: str) -> Dict[str, bool]:
    return {
        "movable": name in MOVABLE,
        "graspable": name in GRASPABLE,
        "placeable": name in PLACEABLE,
        "fragile": name in FRAGILE,
        "electronic": name in FRAGILE,
        "container": name in CONTAINER,
        "liquid_risk": name in LIQUID_RISK,
    }


class SpatialDemoCore:
    # 该函数用于初始化检测与深度模型参数。
    def __init__(self, detector_model: str, depth_model: str, box_thr: float, text_thr: float) -> None:
        self.detector_model = detector_model
        self.depth_model = depth_model
        self.box_thr = float(box_thr)
        self.text_thr = float(text_thr)
        self._detector = None
        self._depth = None

    # 该函数用于确保本地安装了第二阶段所需依赖。
    def ensure_deps(self) -> None:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "缺少依赖 torch/transformers。请先安装后再运行第二阶段（模型下载体积较大，请先确认）。"
            ) from exc

    # 该函数用于懒加载 Grounding DINO 检测管线。
    def detector(self):
        if self._detector is not None:
            return self._detector
        self.ensure_deps()
        from transformers import pipeline

        self._detector = pipeline(task="zero-shot-object-detection", model=self.detector_model)
        return self._detector

    # 该函数用于懒加载 Depth Anything 深度估计管线。
    def depth_pipe(self):
        if self._depth is not None:
            return self._depth
        self.ensure_deps()
        from transformers import pipeline

        self._depth = pipeline(task="depth-estimation", model=self.depth_model)
        return self._depth

    # 该函数用于执行第一步物体检测并返回结构化结果与叠加图。
    def detect_objects(self, image_path: str) -> Dict[str, Any]:
        image = load_rgb(image_path)
        pipe = self.detector()
        raw = pipe(image, candidate_labels=CLASSES, threshold=self.box_thr)

        parsed: List[Dict[str, Any]] = []
        for item in raw:
            name = normalize_label(item.get("label", ""))
            box_raw = item.get("box", {})
            if not isinstance(box_raw, dict):
                continue
            bbox = [
                int(round(float(box_raw.get("xmin", 0)))),
                int(round(float(box_raw.get("ymin", 0)))),
                int(round(float(box_raw.get("xmax", 0)))),
                int(round(float(box_raw.get("ymax", 0)))),
            ]
            bbox = clamp_box(bbox, image.width, image.height)
            score = float(item.get("score", 0.0))
            if score < self.box_thr:
                continue
            parsed.append({"name": name, "bbox": bbox, "score": score})

        parsed = class_nms(parsed, thr=0.55)
        objs = [
            Obj(id=f"obj_{idx + 1}", name=det["name"], bbox=det["bbox"], confidence=round(float(det["score"]), 4))
            for idx, det in enumerate(parsed)
        ]

        overlay = draw_boxes(image=image, objs=objs)
        return {
            "objects": [
                {
                    "id": obj.id,
                    "name": obj.name,
                    "bbox": obj.bbox,
                    "confidence": obj.confidence,
                }
                for obj in objs
            ],
            "overlay_b64": image_b64(overlay),
            "image_size": {"width": image.width, "height": image.height},
        }

    # 该函数用于执行第二步深度估计并返回深度图与每个物体平均深度。
    def estimate_depth(self, image_path: str, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        image = load_rgb(image_path)
        pipe = self.depth_pipe()
        result = pipe(image)

        arr = self.extract_depth(result)
        norm = normalize_map(arr)
        distance = 1.0 - norm
        depth_img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")

        entries: Dict[str, Dict[str, Any]] = {}
        for item in objects:
            oid = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            box = item.get("bbox", [])
            if not oid or not isinstance(box, list) or len(box) != 4:
                continue
            x1, y1, x2, y2 = clamp_box([int(v) for v in box], image.width, image.height)
            roi = distance[y1:y2, x1:x2]
            avg = float(roi.mean()) if roi.size > 0 else float(distance.mean())
            entries[oid] = {"name": name, "avg_depth": round(avg, 4)}

        return {"depth_b64": image_b64(depth_img), "depth_by_object": entries}

    # 该函数用于从深度模型输出中提取二维深度数组。
    def extract_depth(self, result: Any) -> np.ndarray:
        if isinstance(result, dict):
            if "depth" in result and isinstance(result["depth"], Image.Image):
                return np.array(result["depth"]).astype(np.float32)
            if "predicted_depth" in result:
                pred = result["predicted_depth"]
                try:
                    return pred.squeeze().detach().cpu().numpy().astype(np.float32)
                except Exception:
                    pass
        raise RuntimeError("深度模型输出格式无法解析。")

    # 该函数用于执行第三步空间关系推断并输出完整 Scene Graph。
    def build_scene_graph(self, objects: List[Dict[str, Any]], depth_by_object: Dict[str, Any], image_size: Dict[str, Any]) -> Dict[str, Any]:
        if not objects:
            raise ValueError("objects 不能为空。")
        w = int(image_size.get("width", 0))
        h = int(image_size.get("height", 0))
        if w <= 0 or h <= 0:
            raise ValueError("image_size 非法。")

        nodes: List[Obj] = []
        for item in objects:
            oid = str(item.get("id", "")).strip()
            name = normalize_label(str(item.get("name", "")))
            box = item.get("bbox", [])
            conf = float(item.get("confidence", 0.0))
            if not oid or not isinstance(box, list) or len(box) != 4:
                continue
            dep = float(depth_by_object.get(oid, {}).get("avg_depth", 0.5))
            nodes.append(Obj(id=oid, name=name, bbox=clamp_box([int(v) for v in box], w, h), confidence=conf, avg_depth=dep))

        if not nodes:
            raise RuntimeError("objects 无有效内容。")

        relations = self.infer_relations(nodes=nodes, w=w, h=h)
        graph = {
            "objects": [
                {
                    "id": n.id,
                    "name": n.name,
                    "bbox": n.bbox,
                    "avg_depth": round(float(n.avg_depth), 4),
                    "attributes": obj_attrs(n.name),
                }
                for n in nodes
            ],
            "relations": [
                {
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "object": r["object"],
                    "confidence": round(float(r["confidence"]), 4),
                }
                for r in relations
            ],
        }
        return graph

    # 该函数用于根据 bbox 与深度做规则空间关系推断。
    def infer_relations(self, nodes: List[Obj], w: int, h: int) -> List[Dict[str, Any]]:
        rels: List[Dict[str, Any]] = []
        diag = math.sqrt(float(w * w + h * h))
        seen = set()

        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i == j:
                    continue
                ax1, ay1, ax2, ay2 = a.bbox
                bx1, by1, bx2, by2 = b.bbox
                acx, acy = box_center(a.bbox)
                bcx, bcy = box_center(b.bbox)

                # left_of / right_of
                dx = acx - bcx
                if abs(dx) > 12:
                    if dx < 0:
                        self.add_rel(rels, seen, a.name, "left_of", b.name, min(0.95, 0.55 + abs(dx) / max(w, 1)))
                    else:
                        self.add_rel(rels, seen, a.name, "right_of", b.name, min(0.95, 0.55 + abs(dx) / max(w, 1)))

                # above / below
                dy = acy - bcy
                if abs(dy) > 12:
                    if dy < 0:
                        self.add_rel(rels, seen, a.name, "above", b.name, min(0.92, 0.52 + abs(dy) / max(h, 1)))
                    else:
                        self.add_rel(rels, seen, a.name, "below", b.name, min(0.92, 0.52 + abs(dy) / max(h, 1)))

                # near
                gap = box_gap(a.bbox, b.bbox)
                if gap / max(diag, 1.0) < 0.12:
                    self.add_rel(rels, seen, a.name, "near", b.name, min(0.9, 0.75 - gap / max(diag, 1.0)))

                # on + supports
                height_b = max(1, by2 - by1)
                gap_y = abs(ay2 - by1)
                in_x = bx1 <= acx <= bx2
                if in_x and gap_y <= max(10, int(0.12 * height_b)) and ay2 <= by2:
                    self.add_rel(rels, seen, a.name, "on", b.name, 0.78)
                    self.add_rel(rels, seen, b.name, "supports", a.name, 0.78)

                # inside
                if ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2:
                    self.add_rel(rels, seen, a.name, "inside", b.name, 0.73)

                # in_front_of / behind
                dd = float(a.avg_depth - b.avg_depth)
                if abs(dd) > 0.06:
                    if dd < 0:
                        self.add_rel(rels, seen, a.name, "in_front_of", b.name, min(0.95, 0.62 + abs(dd)))
                    else:
                        self.add_rel(rels, seen, a.name, "behind", b.name, min(0.95, 0.62 + abs(dd)))

        return rels

    # 该函数用于写入去重后的关系记录。
    def add_rel(self, rels: List[Dict[str, Any]], seen: set, subj: str, rel: str, obj: str, conf: float) -> None:
        key = (subj, rel, obj)
        if key in seen:
            return
        seen.add(key)
        rels.append({"subject": subj, "relation": rel, "object": obj, "confidence": float(conf)})
