"""
游戏内渲染队列与批处理系统 - 验收测试
运行方式：python -m pytest tests/test_render_queue.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.render_queue import RenderQueue


def make_items():
    return [
        {"id": "a", "mesh": "rock", "material": "stone", "texture": "rock_d",
         "depth": 5.0, "is_static": True, "vertices": 300},
        {"id": "b", "mesh": "rock", "material": "stone", "texture": "rock_d",
         "depth": 3.0, "is_static": True, "vertices": 300},
        {"id": "c", "mesh": "tree", "material": "leaf", "texture": "leaf_d",
         "depth": 8.0, "is_static": True, "vertices": 500},
        {"id": "d", "mesh": "goblin", "material": "skin", "texture": "skin_d",
         "depth": 2.0, "vertices": 200},
        {"id": "e", "mesh": "goblin", "material": "skin", "texture": "skin_d",
         "depth": 4.0, "vertices": 200},
        {"id": "f", "mesh": "goblin", "material": "skin", "texture": "skin_d",
         "depth": 6.0, "vertices": 200},
        {"id": "g", "mesh": "glass", "material": "glass", "texture": "glass_d",
         "depth": 1.0, "transparent": True},
        {"id": "h", "mesh": "glass", "material": "glass", "texture": "glass_d",
         "depth": 9.0, "transparent": True},
    ]


class TestRenderQueue:
    def setup_method(self):
        self.system = RenderQueue()
        self.items = make_items()

    # ---- 渲染队列：排序 + 分层 ----

    def test_case_01(self):
        """不透明层按深度从前到后排序"""
        ordered = self.system.render_queue(self.items)
        opaque = [i for i in ordered if not i.get("transparent")]
        depths = [i["depth"] for i in opaque]
        assert depths == sorted(depths)

    def test_case_02(self):
        """透明层按深度从后到前排序"""
        ordered = self.system.render_queue(self.items)
        transparent = [i for i in ordered if i.get("transparent")]
        depths = [i["depth"] for i in transparent]
        assert depths == sorted(depths, reverse=True)
        assert len(transparent) == 2

    def test_case_03(self):
        """队列分层：不透明层整体排在透明层之前"""
        ordered = self.system.render_queue(self.items)
        flags = [bool(i.get("transparent")) for i in ordered]
        assert flags == sorted(flags)
        assert len(ordered) == len(self.items)

    def test_case_04(self):
        """渲染顺序确定：同一输入多次排序结果一致"""
        first = [i["id"] for i in self.system.render_queue(self.items)]
        for _ in range(5):
            again = [i["id"] for i in self.system.render_queue(self.items)]
            assert again == first

    # ---- 批处理：静态合批 + 动态合批 ----

    def test_case_05(self):
        """静态合批：相同 mesh/material/texture 的静态物体合并为一个批次"""
        batches = self.system.batching(self.items)
        static = [b for b in batches if b["type"] == "static"]
        rock = [b for b in static if b["mesh"] == "rock"]
        assert len(rock) == 1
        assert rock[0]["count"] == 2
        assert {i["id"] for i in rock[0]["items"]} == {"a", "b"}

    def test_case_06(self):
        """动态合批：相同 mesh/material/texture 的动态物体合并为一个批次"""
        batches = self.system.batching(self.items)
        dynamic = [b for b in batches if b["type"] == "dynamic"]
        goblin = [b for b in dynamic if b["mesh"] == "goblin"]
        assert len(goblin) == 1
        assert goblin[0]["count"] == 3

    def test_case_07(self):
        """合批减少 Draw Call：批次数小于物体数"""
        batches = self.system.batching(self.items)
        assert 0 < len(batches) < len(self.items)

    def test_case_08(self):
        """动态合批受顶点上限约束：超过上限拆分为多个批次"""
        system = RenderQueue({"dynamic_batch_vertex_limit": 450})
        items = [
            {"id": str(n), "mesh": "m", "material": "mat", "texture": "t",
             "vertices": 200}
            for n in range(5)
        ]
        batches = system.batching(items)
        assert len(batches) == 3  # 200+200 / 200+200 / 200
        assert all(b["vertices"] <= 450 for b in batches)

    # ---- Draw Call 优化：实例化 + 间接绘制 ----

    def test_case_09(self):
        """GPU Instancing：相同 mesh+material 合并为一次实例化绘制"""
        result = self.system.draw_call_optimization(self.items)
        draws = result["instanced_draws"]
        goblin = [d for d in draws if d["mesh"] == "goblin"]
        assert len(goblin) == 1
        assert goblin[0]["type"] == "instanced"
        assert goblin[0]["instance_count"] == 3

    def test_case_10(self):
        """Draw Indirect：实例化命令打包进间接绘制命令缓冲"""
        result = self.system.draw_call_optimization(self.items)
        indirect = result["indirect"]
        assert indirect["type"] == "draw_indirect"
        assert indirect["command_count"] == len(result["instanced_draws"])
        assert indirect["commands"] == result["instanced_draws"]

    def test_case_11(self):
        """Draw Call 优化减少 Draw Call 数量"""
        result = self.system.draw_call_optimization(self.items)
        assert result["draw_calls_before"] == len(self.items)
        assert result["draw_calls_after"] < result["draw_calls_before"]

    # ---- 状态排序：材质排序 + 纹理排序 ----

    def test_case_12(self):
        """材质排序：同材质对象相邻，减少材质状态切换"""
        ordered = self.system.state_sorting(self.items)
        materials = [i["material"] for i in ordered]
        assert materials == sorted(materials)

    def test_case_13(self):
        """纹理排序：同材质内按纹理排序，减少纹理切换"""
        items = [
            {"id": "1", "material": "m", "texture": "tex_b"},
            {"id": "2", "material": "m", "texture": "tex_a"},
            {"id": "3", "material": "m", "texture": "tex_c"},
        ]
        ordered = self.system.state_sorting(items)
        textures = [i["texture"] for i in ordered]
        assert textures == ["tex_a", "tex_b", "tex_c"]

    # ---- 渲染通道：前向渲染 + 延迟渲染 ----

    def test_case_14(self):
        """渲染通道：不透明走延迟渲染，透明走前向渲染"""
        passes = self.system.render_passes(self.items)
        assert passes["order"] == ["deferred", "forward"]
        deferred = passes["deferred"]
        forward = passes["forward"]
        assert deferred["type"] == "deferred"
        assert forward["type"] == "forward"
        # 延迟通道包含几何与光照两个阶段，且只有不透明物体
        assert len(deferred["geometry"]) == 6
        assert deferred["lighting"] == deferred["geometry"]
        assert all(not i.get("transparent") for i in deferred["geometry"])
        # 前向通道包含全部透明物体
        assert forward["count"] == 2
        assert all(i.get("transparent") for i in forward["items"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
