# 该文件用于实现 Demo 流程中的核心能力：场景分析、场景图构建、动作计划与计划校验。
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List
from urllib import error, request


ALLOWED_ACTIONS = {
    "navigate",
    "detect",
    "grasp",
    "move",
    "place",
    "verify",
    "avoid",
}


# 该函数用于去除模型输出中的 Markdown 代码块包裹。
def strip_code_fence(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return raw


# 该函数用于从混合文本中提取首个 JSON 对象或数组。
def extract_json_text(text: str) -> str | None:
    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]
    start_arr = text.find("[")
    end_arr = text.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]
    return None


# 该函数用于严格解析模型返回的 JSON，解析失败直接报错。
def parse_json_strict(text: str) -> Any:
    clean = strip_code_fence(text)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        chunk = extract_json_text(clean)
        if chunk is None:
            preview = clean[:240].replace("\n", " ")
            raise RuntimeError(f"模型返回不是合法 JSON，内容片段: {preview}")
        try:
            return json.loads(chunk)
        except json.JSONDecodeError as exc:
            preview = clean[:240].replace("\n", " ")
            raise RuntimeError(f"模型 JSON 解析失败: {exc}. 内容片段: {preview}") from exc


# 该函数用于把本地图片编码为 data URL。
def encode_image(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


# 该函数用于把对象标签规范成可读 ID。
def make_oid(label: str, index: int) -> str:
    safe = label.strip().lower().replace(" ", "_").replace("-", "_")
    safe = "".join(ch for ch in safe if ch.isalnum() or ch == "_").strip("_")
    if not safe:
        safe = "object"
    return f"{safe}_{index + 1}"


# 该函数用于把任意对象值标准化为字符串列表。
def to_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


class SpatialDemoCore:
    # 该函数用于初始化模型调用参数。
    def __init__(self, api_base: str, api_key: str, model: str, timeout: int) -> None:
        if not api_base:
            raise ValueError("api_base 不能为空。")
        if not api_key:
            raise ValueError("api_key 不能为空。")
        if not model:
            raise ValueError("model 不能为空。")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = int(timeout)

    # 该函数用于调用 OpenAI 兼容接口并返回纯文本内容。
    def chat(self, payload: Dict[str, Any]) -> str:
        endpoint = f"{self.api_base}/chat/completions"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"模型接口调用失败: {exc.code} {text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"模型接口网络错误: {exc.reason}") from exc

        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("模型接口返回为空。")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    chunks.append(str(part.get("text", "")))
            text = "\n".join(chunks).strip()
            if text:
                return text
        raise RuntimeError("模型返回内容格式不支持。")

    # 该函数用于调用模型做场景理解，输出描述、物体与关系。
    def analyze_scene(self, image_path: str) -> Dict[str, Any]:
        image_url = encode_image(image_path)
        payload = self.build_analyze_payload(image_url=image_url)
        raw = self.chat(payload)
        parsed = parse_json_strict(raw)
        return self.normalize_analysis(parsed)

    # 该函数用于基于图片、任务和上下文（分析结果或场景图）生成动作计划。
    def generate_action_plan(self, image_path: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if not task.strip():
            raise ValueError("任务不能为空。")
        if not isinstance(context, dict) or not context:
            raise ValueError("动作计划需要 context（analysis 或 scene_graph）。")
        image_url = encode_image(image_path)
        payload = self.build_plan_payload(image_url=image_url, task=task, context=context)
        raw = self.chat(payload)
        parsed = parse_json_strict(raw)
        return self.normalize_plan(parsed)

    # 该函数用于构建场景分析请求。
    def build_analyze_payload(self, image_url: str) -> Dict[str, Any]:
        instruction = (
            "Analyze this indoor scene and return JSON only. "
            "Schema: {"
            "\"room_description\":\"...\","
            "\"objects\":[{\"id\":\"mug_1\",\"label\":\"mug\",\"attributes\":[\"graspable\"]}],"
            "\"relations\":[{\"source\":\"mug_1\",\"relation\":\"on\",\"target\":\"coffee_table_1\"}]"
            "}. "
            "Use concise English. Keep id stable and object-centric."
        )
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a spatial scene parser for robotics."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }

    # 该函数用于构建动作计划请求。
    def build_plan_payload(self, image_url: str, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        context_text = json.dumps(context, ensure_ascii=False)
        instruction = (
            "You are a robot task planner. Return JSON only. "
            "Schema: {"
            "\"plan_summary\":\"...\","
            "\"plan_steps\":["
            "{\"step\":1,\"action\":\"navigate\",\"target\":\"coffee_table_1\",\"reference\":\"\",\"description\":\"Navigate to the coffee table\"}"
            "]}. "
            "Allowed actions: navigate, detect, grasp, move, place, verify, avoid. "
            "Use object ids from context when possible."
        )
        user_text = f"Task: {task}\nContext: {context_text}"
        return {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You generate executable robot plans from images and scene graphs."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }

    # 该函数用于把模型分析结果标准化并做严格校验。
    def normalize_analysis(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("场景分析返回格式错误：必须是 JSON 对象。")
        room = str(payload.get("room_description", payload.get("description", ""))).strip()
        if not room:
            raise RuntimeError("场景分析缺少 room_description。")

        raw_objects = payload.get("objects", payload.get("items", []))
        if not isinstance(raw_objects, list) or not raw_objects:
            raise RuntimeError("场景分析缺少 objects。")
        objects: List[Dict[str, Any]] = []
        label_to_id: Dict[str, str] = {}

        for idx, item in enumerate(raw_objects):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", item.get("name", ""))).strip().lower()
            if not label:
                continue
            oid = str(item.get("id", item.get("oid", ""))).strip().lower()
            if not oid:
                oid = make_oid(label, idx)
            attrs = to_string_list(item.get("attributes", item.get("affordance", [])))
            obj = {"id": oid, "label": label, "attributes": attrs}
            objects.append(obj)
            label_to_id[label] = oid

        if not objects:
            raise RuntimeError("场景分析 objects 无有效内容。")

        raw_relations = payload.get("relations", payload.get("triples", []))
        relations: List[Dict[str, str]] = []
        if isinstance(raw_relations, list):
            for item in raw_relations:
                norm = self.normalize_relation(item=item, label_to_id=label_to_id)
                if norm is not None:
                    relations.append(norm)

        if not relations:
            raise RuntimeError("场景分析缺少 relations。")

        relations_text = [f"{r['source']} -> {r['relation']} -> {r['target']}" for r in relations]
        return {
            "room_description": room,
            "objects": objects,
            "relations": relations,
            "relations_text": relations_text,
        }

    # 该函数用于标准化单条关系三元组。
    def normalize_relation(self, item: Any, label_to_id: Dict[str, str]) -> Dict[str, str] | None:
        if not isinstance(item, dict):
            return None
        source = str(item.get("source", item.get("from", ""))).strip().lower()
        relation = str(item.get("relation", item.get("predicate", ""))).strip().lower()
        target = str(item.get("target", item.get("to", ""))).strip().lower()
        if not source or not relation or not target:
            return None

        source = label_to_id.get(source, source)
        target = label_to_id.get(target, target)
        return {"source": source, "relation": relation, "target": target}

    # 该函数用于把模型计划结果标准化并做严格校验。
    def normalize_plan(self, payload: Any) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError("动作计划返回格式错误：必须是 JSON 对象。")
        summary = str(payload.get("plan_summary", payload.get("summary", ""))).strip()
        raw_steps = payload.get("plan_steps", payload.get("steps", []))
        if not isinstance(raw_steps, list) or not raw_steps:
            raise RuntimeError("动作计划缺少 plan_steps。")

        plan_steps: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_steps):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip().lower()
            if action not in ALLOWED_ACTIONS:
                raise RuntimeError(f"动作计划包含不支持的 action: {action}")
            target = str(item.get("target", "")).strip().lower()
            reference = str(item.get("reference", "")).strip().lower()
            desc = str(item.get("description", item.get("note", ""))).strip()
            step_no = int(item.get("step", idx + 1))
            plan_steps.append(
                {
                    "step": step_no,
                    "action": action,
                    "target": target,
                    "reference": reference,
                    "description": desc or action,
                }
            )

        if not plan_steps:
            raise RuntimeError("动作计划 plan_steps 无有效内容。")
        return {"plan_summary": summary, "plan_steps": plan_steps}
