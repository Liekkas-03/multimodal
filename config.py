# 该文件用于加载并解析项目配置文件。

import tomllib
from pathlib import Path
from typing import Any, Dict


# 该函数用于读取 TOML 配置文件并返回字典结构。
def load_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    file = Path(path)
    if not file.exists():
        return {}
    with file.open("rb") as f:
        data = tomllib.load(f)
    if isinstance(data, dict):
        return data
    return {}


# 该函数用于按“节+键”读取配置项并返回默认值兜底。
def get_config_value(config: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    group = config.get(section, {})
    if not isinstance(group, dict):
        return default
    return group.get(key, default)
