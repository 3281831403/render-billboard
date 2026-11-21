# render-billboard

## 题面

粒子公告板换成朝向相机后，角色一低头贴图就翻、近处还消失：叉积顺序反了，法线朝内；不归一化基向量，缩放任距离漂移；yaw 模式也全轴朝向，角色倾斜；屏幕缩放按世界尺寸，远近一样大；上方向不对齐，相机俯仰时翻转；轴向锁定不绕 Y，物体绕错轴；深度衰减在近处为 0，贴脸消失；模式未知静默当 full；批量共用基向量，实例之间串；相机与物体重合时除零。billboard/facing.py 是核心，tests/test_facing.py 15个红。要修的是：叉积右手方向与法线朝外、基向量归一化、yaw 仅绕 Y 保持直立、透视缩放随距离收缩、相机上方向正交对齐、轴向锁定绕 Y、近处深度衰减为 1 且远处不消失、未知模式抛 BillboardError、批量实例独立基向量与任一非法实例整体回滚、相机与物体重合时抛 BillboardError。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始失败环境预期恰好为 `15 failed`、`0 error`，且没有 collection、import 或 syntax error：

```text
15 failed
```

15 条测试全部因桩函数 `raise NotImplementedError` 而失败；实现契约后应全部通过。测试只使用标准库 `math` 与 pytest，离线运行、无随机性，数值断言容差为 `1e-6`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- `billboard/facing.py` 中的 8 组函数（`cross` 与 `normalize` 合为向量工具组，共 9 个 callable）全部为 `raise NotImplementedError` 桩，需按 docstring 契约实现；异常统一使用 `BillboardError`。
- 相机统一使用 `{"pos", "up", "forward"}`，基向量统一为 `(right, up, normal)`，朝向模式为 `full`、`yaw`、`axial`，缩放模式为 `perspective`、`screen`、`world`。
- `tests/test_facing.py` 是验收测试，不得修改、删除或新增；修复只能落在 `billboard/facing.py`。
- 仅使用 Python 标准库 `math` 与 pytest；不得引入网络、随机数或其他外部依赖。
