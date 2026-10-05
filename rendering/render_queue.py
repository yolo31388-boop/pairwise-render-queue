"""
游戏内渲染队列与批处理系统 - 核心模块

功能：
- render_queue():            渲染队列（排序 + 不透明/透明分层）
- batching():                批处理（静态合批 + 动态合批）
- draw_call_optimization():  Draw Call 优化（GPU Instancing + Draw Indirect）
- state_sorting():           状态排序（材质排序 + 纹理排序）
- render_passes():           渲染通道（前向渲染 + 延迟渲染）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


# 渲染层
LAYER_OPAQUE = "opaque"
LAYER_TRANSPARENT = "transparent"

# 渲染通道
PASS_FORWARD = "forward"
PASS_DEFERRED = "deferred"

# 动态合批的顶点数上限（单个动态批次累计顶点不超过该值）
DYNAMIC_BATCH_VERTEX_LIMIT = 1024


@dataclass
class Config:
    dynamic_batch_vertex_limit: int = DYNAMIC_BATCH_VERTEX_LIMIT


def _depth(item: Dict) -> float:
    return float(item.get("depth", item.get("z", 0.0)))


def _layer_of(item: Dict) -> str:
    layer = item.get("layer")
    if layer in (LAYER_OPAQUE, LAYER_TRANSPARENT):
        return layer
    return LAYER_TRANSPARENT if item.get("transparent") else LAYER_OPAQUE


def _sort_key(item: Dict):
    return (
        int(item.get("render_order", 0)),
        str(item.get("material", "")),
        str(item.get("texture", "")),
        str(item.get("mesh", "")),
        str(item.get("id", "")),
    )


class RenderQueue:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._queue: List[Dict] = []

    def update(self, dt: float):
        self._state["last_update"] = time.time()
        self._state["dt"] = dt

    def reset(self):
        self._state = {}
        self._history = []
        self._queue = []

    # ------------------------------------------------------------------
    # 提交
    # ------------------------------------------------------------------
    def submit(self, item: Dict):
        """提交一个可渲染对象到队列。"""
        self._queue.append(dict(item))

    def clear(self):
        self._queue = []

    def _items(self, items: Optional[List[Dict]]) -> List[Dict]:
        return [dict(i) for i in (items if items is not None else self._queue)]

    # ------------------------------------------------------------------
    # 渲染队列：排序 + 分层
    # ------------------------------------------------------------------
    def render_queue(self, items: Optional[List[Dict]] = None) -> List[Dict]:
        """
        构建有序渲染队列：
        - 分层：不透明层在前，透明层在后；
        - 不透明层按深度从前到后排序（减少 overdraw）；
        - 透明层按深度从后到前排序（保证混合正确）；
        - 排序结果确定（稳定键），不会随机。
        """
        queued = self._items(items)
        opaque = [i for i in queued if _layer_of(i) == LAYER_OPAQUE]
        transparent = [i for i in queued if _layer_of(i) == LAYER_TRANSPARENT]

        opaque.sort(key=lambda i: (_depth(i),) + _sort_key(i))
        transparent.sort(key=lambda i: (-_depth(i),) + _sort_key(i))

        ordered = opaque + transparent
        self._history.append({"op": "render_queue", "count": len(ordered)})
        return ordered

    # ------------------------------------------------------------------
    # 批处理：静态合批 + 动态合批
    # ------------------------------------------------------------------
    def batching(self, items: Optional[List[Dict]] = None) -> List[Dict]:
        """
        将可渲染对象合批：
        - 静态合批：is_static 为真且 (mesh, material, texture) 相同的对象
          合并为一个静态批次；
        - 动态合批：动态对象按 (mesh, material, texture) 分组，在单批次
          累计顶点数不超过上限的前提下合并；
        返回批次列表，每个批次对应一次 Draw Call。
        """
        vertex_limit = int(
            self.config.get("dynamic_batch_vertex_limit", DYNAMIC_BATCH_VERTEX_LIMIT)
        )
        queued = self.render_queue(items)

        static_groups: Dict[Tuple, List[Dict]] = {}
        dynamic_groups: Dict[Tuple, List[Dict]] = {}
        for item in queued:
            key = (
                item.get("mesh"),
                item.get("material"),
                item.get("texture"),
            )
            if item.get("is_static"):
                static_groups.setdefault(key, []).append(item)
            else:
                dynamic_groups.setdefault(key, []).append(item)

        batches: List[Dict] = []
        for key, members in static_groups.items():
            batches.append({
                "type": "static",
                "mesh": key[0],
                "material": key[1],
                "texture": key[2],
                "items": members,
                "count": len(members),
                "vertices": sum(int(m.get("vertices", 0)) for m in members),
            })

        for key, members in dynamic_groups.items():
            current: List[Dict] = []
            current_vertices = 0
            for item in members:
                vertices = int(item.get("vertices", 0))
                if current and current_vertices + vertices > vertex_limit:
                    batches.append(self._make_dynamic_batch(key, current, current_vertices))
                    current, current_vertices = [], 0
                current.append(item)
                current_vertices += vertices
            if current:
                batches.append(self._make_dynamic_batch(key, current, current_vertices))

        self._history.append({
            "op": "batching",
            "draw_calls_before": len(queued),
            "draw_calls_after": len(batches),
        })
        return batches

    @staticmethod
    def _make_dynamic_batch(key: Tuple, members: List[Dict], vertices: int) -> Dict:
        return {
            "type": "dynamic",
            "mesh": key[0],
            "material": key[1],
            "texture": key[2],
            "items": list(members),
            "count": len(members),
            "vertices": vertices,
        }

    # ------------------------------------------------------------------
    # Draw Call 优化：GPU Instancing + Draw Indirect
    # ------------------------------------------------------------------
    def draw_call_optimization(self, items: Optional[List[Dict]] = None) -> Dict:
        """
        Draw Call 优化：
        - GPU Instancing：(mesh, material) 相同的对象合并为一次实例化绘制；
        - Draw Indirect：所有实例化绘制命令打包进一个间接绘制命令缓冲，
          由一次 Draw Indirect 提交。
        返回包含实例化绘制列表、间接绘制命令与 Draw Call 统计的字典。
        """
        queued = self.render_queue(items)

        groups: Dict[Tuple, List[Dict]] = {}
        for item in queued:
            key = (item.get("mesh"), item.get("material"))
            groups.setdefault(key, []).append(item)

        instanced_draws = []
        for (mesh, material), members in groups.items():
            instanced_draws.append({
                "type": "instanced",
                "mesh": mesh,
                "material": material,
                "instances": members,
                "instance_count": len(members),
            })

        indirect = {
            "type": "draw_indirect",
            "commands": instanced_draws,
            "command_count": len(instanced_draws),
        }

        result = {
            "instanced_draws": instanced_draws,
            "indirect": indirect,
            "draw_calls_before": len(queued),
            "draw_calls_after": len(instanced_draws),
        }
        self._history.append({
            "op": "draw_call_optimization",
            "draw_calls_before": result["draw_calls_before"],
            "draw_calls_after": result["draw_calls_after"],
        })
        return result

    # ------------------------------------------------------------------
    # 状态排序：材质排序 + 纹理排序
    # ------------------------------------------------------------------
    def state_sorting(self, items: Optional[List[Dict]] = None) -> List[Dict]:
        """
        按渲染状态排序，减少状态切换：
        - 先按材质排序（减少 shader/管线状态切换）；
        - 材质相同再按纹理排序（减少纹理绑定切换）；
        - 排序键稳定，结果确定。
        """
        queued = self._items(items)
        ordered = sorted(
            queued,
            key=lambda i: (
                str(i.get("material", "")),
                str(i.get("texture", "")),
                str(i.get("mesh", "")),
                str(i.get("id", "")),
            ),
        )
        self._history.append({"op": "state_sorting", "count": len(ordered)})
        return ordered

    # ------------------------------------------------------------------
    # 渲染通道：前向渲染 + 延迟渲染
    # ------------------------------------------------------------------
    def render_passes(self, items: Optional[List[Dict]] = None) -> Dict:
        """
        将渲染对象分配到渲染通道：
        - 延迟渲染通道：不透明对象，先几何阶段写入 G-Buffer，再光照阶段；
        - 前向渲染通道：透明对象及显式指定 forward 的对象，直接着色输出；
        延迟通道先于前向通道执行。
        """
        queued = self.render_queue(items)

        deferred_items = []
        forward_items = []
        for item in queued:
            requested = item.get("pass")
            if requested == PASS_FORWARD or (
                requested is None and _layer_of(item) == LAYER_TRANSPARENT
            ):
                forward_items.append(item)
            else:
                deferred_items.append(item)

        passes = {
            "deferred": {
                "type": PASS_DEFERRED,
                "geometry": list(deferred_items),
                "lighting": list(deferred_items),
                "count": len(deferred_items),
            },
            "forward": {
                "type": PASS_FORWARD,
                "items": forward_items,
                "count": len(forward_items),
            },
            "order": [PASS_DEFERRED, PASS_FORWARD],
        }
        self._history.append({
            "op": "render_passes",
            "deferred": len(deferred_items),
            "forward": len(forward_items),
        })
        return passes
