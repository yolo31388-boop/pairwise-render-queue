"""
游戏内渲染队列与批处理系统 - 核心模块

功能：
- render_queue():           渲染队列（排序 + 不透明/透明分层）
- batching():               批处理（静态合批 + 动态合批）
- draw_call_optimization(): Draw Call 优化（GPU Instancing + Draw Indirect）
- state_sorting():          状态排序（材质排序 + 纹理排序）
- render_passes():          渲染通道（前向渲染 + 延迟渲染）
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


# 动态合批的顶点数上限（参考 Unity 的动态合批阈值）
DYNAMIC_BATCH_VERTEX_LIMIT = 300


@dataclass
class DrawItem:
    """一个待渲染对象的描述。"""
    id: int
    mesh_id: int
    material_id: int
    texture_id: int
    depth: float = 0.0
    transparent: bool = False
    static: bool = False
    vertex_count: int = 0


class RenderQueue:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._submitted: List[DrawItem] = []

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._submitted = []

    # ------------------------------------------------------------------
    # 提交接口
    # ------------------------------------------------------------------
    def submit(self, item) -> None:
        """提交一个渲染对象（DrawItem 或 dict）。"""
        self._submitted.append(self._normalize(item))

    def clear(self) -> None:
        self._submitted = []

    @staticmethod
    def _normalize(item) -> DrawItem:
        if isinstance(item, DrawItem):
            return item
        if isinstance(item, dict):
            return DrawItem(**item)
        raise TypeError("item 必须是 DrawItem 或 dict")

    def _resolve(self, items=None) -> List[DrawItem]:
        if items is None:
            return list(self._submitted)
        return [self._normalize(i) for i in items]

    # ------------------------------------------------------------------
    # 渲染队列：排序 + 分层
    # ------------------------------------------------------------------
    def render_queue(self, items=None) -> Dict:
        """
        构建渲染队列：
        - 分层：不透明层（opaque）与透明层（transparent）分开；
        - 排序：不透明层按深度从前到后（利于 Early-Z），
                透明层按深度从后到前（保证混合正确）；
        - 确定性：以 id 作为最终 tiebreaker，同样的输入永远得到同样的顺序。
        """
        items = self._resolve(items)
        opaque = [i for i in items if not i.transparent]
        transparent = [i for i in items if i.transparent]
        opaque.sort(key=lambda i: (i.depth, i.material_id, i.texture_id, i.id))
        transparent.sort(key=lambda i: (-i.depth, i.material_id, i.texture_id, i.id))
        return {
            "opaque": opaque,
            "transparent": transparent,
            "order": opaque + transparent,
        }

    # ------------------------------------------------------------------
    # 批处理：静态合批 + 动态合批
    # ------------------------------------------------------------------
    def batching(self, items=None) -> Dict:
        """
        合批：
        - 静态合批：static 物体按材质合并进大顶点缓冲，每组 1 个 Draw Call；
        - 动态合批：顶点数不超过阈值的运动物体按材质合并，每组 1 个 Draw Call；
        - 其余（顶点数过多的动态物体）无法合批，逐个绘制。
        """
        items = self._resolve(items)
        batches: List[Dict] = []

        static_groups: Dict[int, List[DrawItem]] = {}
        dynamic_groups: Dict[int, List[DrawItem]] = {}
        individuals: List[DrawItem] = []

        for item in items:
            if item.static:
                static_groups.setdefault(item.material_id, []).append(item)
            elif item.vertex_count <= DYNAMIC_BATCH_VERTEX_LIMIT:
                dynamic_groups.setdefault(item.material_id, []).append(item)
            else:
                individuals.append(item)

        for material_id in sorted(static_groups):
            group = static_groups[material_id]
            batches.append({
                "type": "static",
                "material_id": material_id,
                "items": group,
                "draw_calls": 1,
            })
        for material_id in sorted(dynamic_groups):
            group = dynamic_groups[material_id]
            batches.append({
                "type": "dynamic",
                "material_id": material_id,
                "items": group,
                "draw_calls": 1,
            })
        for item in individuals:
            batches.append({
                "type": "individual",
                "material_id": item.material_id,
                "items": [item],
                "draw_calls": 1,
            })

        draw_calls = sum(b["draw_calls"] for b in batches)
        return {
            "batches": batches,
            "draw_calls": draw_calls,
            "original_draw_calls": len(items),
        }

    # ------------------------------------------------------------------
    # Draw Call 优化：GPU Instancing + Draw Indirect
    # ------------------------------------------------------------------
    def draw_call_optimization(self, items=None) -> Dict:
        """
        - GPU Instancing：相同 (mesh, material) 的物体合并为一次实例化绘制；
        - Draw Indirect：把实例化组打包成间接绘制命令缓冲，
          由 GPU 自行展开，CPU 侧只需一次提交。
        """
        items = self._resolve(items)
        groups: Dict[Tuple[int, int], List[DrawItem]] = {}
        for item in items:
            groups.setdefault((item.mesh_id, item.material_id), []).append(item)

        instanced_groups = []
        indirect_commands = []
        first_instance = 0
        for (mesh_id, material_id) in sorted(groups):
            group = groups[(mesh_id, material_id)]
            instanced_groups.append({
                "mesh_id": mesh_id,
                "material_id": material_id,
                "instances": group,
                "instance_count": len(group),
                "draw_calls": 1,
            })
            indirect_commands.append({
                "mesh_id": mesh_id,
                "material_id": material_id,
                "instance_count": len(group),
                "first_instance": first_instance,
            })
            first_instance += len(group)

        return {
            "instanced_groups": instanced_groups,
            "indirect_commands": indirect_commands,
            "draw_calls": len(instanced_groups),
            "original_draw_calls": len(items),
        }

    # ------------------------------------------------------------------
    # 状态排序：材质排序 + 纹理排序
    # ------------------------------------------------------------------
    def state_sorting(self, items=None) -> Dict:
        """
        按 (材质, 纹理) 排序，把相同渲染状态的对象排在一起，
        最大限度减少材质/纹理状态切换。
        """
        items = self._resolve(items)
        ordered = sorted(
            items,
            key=lambda i: (i.material_id, i.texture_id, i.depth, i.id),
        )
        material_switches = 0
        texture_switches = 0
        prev = None
        for item in ordered:
            key = (item.material_id, item.texture_id)
            if prev is not None and key != prev:
                if item.material_id != prev[0]:
                    material_switches += 1
                texture_switches += 1
            prev = key
        return {
            "order": ordered,
            "material_switches": material_switches,
            "texture_switches": texture_switches,
            "state_switches": texture_switches,
        }

    # ------------------------------------------------------------------
    # 渲染通道：前向渲染 + 延迟渲染
    # ------------------------------------------------------------------
    def render_passes(self, items=None, mode: str = "forward") -> Dict:
        """
        - forward：不透明通道 + 透明通道，逐物体光照；
        - deferred：几何通道（仅不透明，写 G-Buffer）
                    + 光照通道（一次全屏绘制，与光源数无关）
                    + 透明通道（透明物体仍走前向）。
        """
        queue = self.render_queue(items)
        opaque = queue["opaque"]
        transparent = queue["transparent"]

        if mode == "forward":
            passes = [
                {"name": "opaque", "items": opaque, "draw_calls": len(opaque)},
                {"name": "transparent", "items": transparent,
                 "draw_calls": len(transparent)},
            ]
        elif mode == "deferred":
            passes = [
                {"name": "geometry", "items": opaque, "draw_calls": len(opaque)},
                {"name": "lighting", "items": [], "draw_calls": 1},
                {"name": "transparent", "items": transparent,
                 "draw_calls": len(transparent)},
            ]
        else:
            raise ValueError("未知渲染模式: %r" % (mode,))

        return {
            "mode": mode,
            "passes": passes,
            "draw_calls": sum(p["draw_calls"] for p in passes),
        }
