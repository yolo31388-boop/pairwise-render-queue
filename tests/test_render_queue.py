"""
游戏内渲染队列与批处理系统 - 验收测试
运行方式：python -m pytest tests/test_render_queue.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.render_queue import RenderQueue, DrawItem


def make_items():
    """构造一组混合的渲染对象：不透明/透明、静态/动态、不同材质纹理。"""
    return [
        DrawItem(id=1, mesh_id=10, material_id=2, texture_id=20,
                 depth=5.0, static=True, vertex_count=500),
        DrawItem(id=2, mesh_id=10, material_id=2, texture_id=20,
                 depth=8.0, static=True, vertex_count=500),
        DrawItem(id=3, mesh_id=11, material_id=1, texture_id=10,
                 depth=2.0, vertex_count=100),
        DrawItem(id=4, mesh_id=11, material_id=1, texture_id=10,
                 depth=3.0, vertex_count=100),
        DrawItem(id=5, mesh_id=12, material_id=1, texture_id=11,
                 depth=4.0, vertex_count=100),
        DrawItem(id=6, mesh_id=13, material_id=3, texture_id=30,
                 depth=1.0, vertex_count=5000),
        DrawItem(id=7, mesh_id=14, material_id=4, texture_id=40,
                 depth=6.0, transparent=True, vertex_count=50),
        DrawItem(id=8, mesh_id=14, material_id=4, texture_id=40,
                 depth=9.0, transparent=True, vertex_count=50),
    ]


class TestRenderQueue:
    def setup_method(self):
        self.system = RenderQueue()

    # ---- 渲染队列：分层 + 排序 ----

    def test_case_01(self):
        """渲染队列分层：不透明与透明对象被分到不同的层。"""
        result = self.system.render_queue(make_items())
        assert all(not i.transparent for i in result["opaque"])
        assert all(i.transparent for i in result["transparent"])
        assert len(result["opaque"]) == 6
        assert len(result["transparent"]) == 2

    def test_case_02(self):
        """渲染队列排序：不透明层按深度从前到后排序。"""
        result = self.system.render_queue(make_items())
        depths = [i.depth for i in result["opaque"]]
        assert depths == sorted(depths)

    def test_case_03(self):
        """渲染队列排序：透明层按深度从后到前排序。"""
        result = self.system.render_queue(make_items())
        depths = [i.depth for i in result["transparent"]]
        assert depths == sorted(depths, reverse=True)

    def test_case_04(self):
        """渲染队列确定性：同样输入多次构建，顺序完全一致（不随机）。"""
        items = make_items()
        first = [i.id for i in self.system.render_queue(items)["order"]]
        for _ in range(5):
            again = [i.id for i in self.system.render_queue(items)["order"]]
            assert again == first

    # ---- 批处理：静态合批 + 动态合批 ----

    def test_case_05(self):
        """静态合批：同材质的静态物体合并为一个批次、一次 Draw Call。"""
        result = self.system.batching(make_items())
        static_batches = [b for b in result["batches"] if b["type"] == "static"]
        assert len(static_batches) == 1
        assert static_batches[0]["draw_calls"] == 1
        assert {i.id for i in static_batches[0]["items"]} == {1, 2}

    def test_case_06(self):
        """动态合批：顶点数低于阈值的动态物体按材质合并。"""
        result = self.system.batching(make_items())
        dynamic_batches = [b for b in result["batches"] if b["type"] == "dynamic"]
        assert len(dynamic_batches) >= 1
        batched_ids = {i.id for b in dynamic_batches for i in b["items"]}
        assert {3, 4}.issubset(batched_ids)
        for batch in dynamic_batches:
            assert batch["draw_calls"] == 1
            for item in batch["items"]:
                assert not item.static
                assert item.vertex_count <= 300

    def test_case_07(self):
        """合批收益：合批后的 Draw Call 数显著少于原始对象数。"""
        items = make_items()
        result = self.system.batching(items)
        assert result["original_draw_calls"] == len(items)
        assert result["draw_calls"] < result["original_draw_calls"]

    # ---- Draw Call 优化：实例化 + 间接绘制 ----

    def test_case_08(self):
        """GPU Instancing：相同 mesh+material 的对象合并为一次实例化绘制。"""
        result = self.system.draw_call_optimization(make_items())
        groups = {(g["mesh_id"], g["material_id"]): g
                  for g in result["instanced_groups"]}
        assert (10, 2) in groups
        assert groups[(10, 2)]["instance_count"] == 2
        assert groups[(10, 2)]["draw_calls"] == 1
        assert (14, 4) in groups
        assert groups[(14, 4)]["instance_count"] == 2

    def test_case_09(self):
        """Draw Indirect：生成间接绘制命令缓冲，实例偏移连续无重叠。"""
        result = self.system.draw_call_optimization(make_items())
        commands = result["indirect_commands"]
        assert len(commands) == len(result["instanced_groups"])
        offset = 0
        for cmd in commands:
            assert cmd["first_instance"] == offset
            assert cmd["instance_count"] > 0
            offset += cmd["instance_count"]
        assert offset == len(make_items())

    def test_case_10(self):
        """优化收益：实例化后的 Draw Call 数少于原始对象数。"""
        items = make_items()
        result = self.system.draw_call_optimization(items)
        assert result["original_draw_calls"] == len(items)
        assert result["draw_calls"] < result["original_draw_calls"]

    # ---- 状态排序：材质排序 + 纹理排序 ----

    def test_case_11(self):
        """材质排序：排序后相同材质的对象相邻，材质切换次数最少。"""
        result = self.system.state_sorting(make_items())
        materials = [i.material_id for i in result["order"]]
        assert materials == sorted(materials)
        assert result["material_switches"] == len(set(materials)) - 1

    def test_case_12(self):
        """纹理排序：同一材质内按纹理排序，减少纹理切换。"""
        result = self.system.state_sorting(make_items())
        keys = [(i.material_id, i.texture_id) for i in result["order"]]
        assert keys == sorted(keys)
        assert result["texture_switches"] == len(set(keys)) - 1

    # ---- 渲染通道：前向渲染 + 延迟渲染 ----

    def test_case_13(self):
        """前向渲染：分不透明/透明两个通道，覆盖全部对象。"""
        items = make_items()
        result = self.system.render_passes(items, mode="forward")
        names = [p["name"] for p in result["passes"]]
        assert names == ["opaque", "transparent"]
        total = sum(len(p["items"]) for p in result["passes"])
        assert total == len(items)
        assert result["draw_calls"] == len(items)

    def test_case_14(self):
        """延迟渲染：几何通道只含不透明物体，光照通道固定 1 次全屏绘制。"""
        items = make_items()
        result = self.system.render_passes(items, mode="deferred")
        passes = {p["name"]: p for p in result["passes"]}
        assert set(passes) == {"geometry", "lighting", "transparent"}
        assert all(not i.transparent for i in passes["geometry"]["items"])
        assert passes["lighting"]["draw_calls"] == 1
        assert len(passes["transparent"]["items"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
