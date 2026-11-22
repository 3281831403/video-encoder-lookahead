# video-encoder-lookahead

编码器加前视分析后，场景切换处还是不出 I 帧、码率还超预算：`plan_gop` 用 `<=` 判断 I 间隔，最后一个 GOP 可能超 keyint 一帧；场景切分点只在 P 帧上生效，落在 B 帧槽位的切分被忽略；`assign_bits` 先给 I 帧分配再整体归一化，归一化把 I 帧的权重一起缩掉，I 帧不再显著高于 P 帧；`assign_bits` 直接拿复杂度原始值加权，总比特随复杂度量级漂移；`reorder_bframes` 把连续 B 帧重排成逆序，编码序错误；`reorder_bframes` 在 GOP 边界把最后一个 B 与下一个 I 交换，参考链断裂；`budget_check` 用 `abs(diff) < 1` 判 ok，一位容差的边界被误杀；`validate_plan` 不检查 B 帧是否引用了未来的 I/P，非法参考链漏检；`scene_scores` 的窗口包含当前帧，切分点判断滞后一帧；`plan_batch` 复用 complexity 数组，第二片沿用第一片的复杂度。难点是规划守恒：I 间隔、总预算、参考链三处必须同时成立。lookahead/plan.py 是核心，tests/test_plan.py 15个红。要修的是：实现全部 8 个桩函数，逐条消除上述规划/预算缺陷，让 15 个测试在离线、无随机、仅标准库 `math` + `pytest` 的环境下全绿，且不得修改 `tests/`。

## 运行方式

在仓库根目录执行：

```bash
python -m pytest tests -q
```

初始（8 个桩函数都未实现）的输出必须恰好是：

```text
15 failed in 0.11s
```

耗时因机器而异，判定标准是恰好 `15 failed`，且没有 collection / import / syntax error；15 条测试全部因 `NotImplementedError` 变红。完成实现后同一条命令应当是 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只允许 Python 标准库（`math`）与 `pytest`；离线运行，不得联网，不得使用随机数、当前时间或外部文件参与规划。
- `tests/test_plan.py` 不可修改：它是验收规格，15 条测试覆盖 10 个规划/预算缺陷，以及 I 间隔、预算守恒、I 帧比特优势、参考链、批量独立性 5 条性质。改测试视为未完成。
- 保持 8 个公开函数的签名与返回形状；`lookahead/plan.py` 中常量的语义也是契约的一部分。

## 文件结构

```text
video-encoder-lookahead/
├── README.md
├── conftest.py
├── .gitignore
├── lookahead/
│   ├── __init__.py
│   └── plan.py
└── tests/
    └── test_plan.py
```

## 接口契约

| 函数 | 职责 |
| --- | --- |
| `complexity(frames)` | 帧间差 + 纹理方差代理，归一化到 `[0, 1]` |
| `scene_scores(complexity, window)` | 严格因果滑窗归一化，供切分判定 |
| `plan_gop(n_frames, keyint, scene_cuts)` | 生成显示序 I/P/B，满足 keyint 与场景切分 |
| `assign_bits(types, budget, complexity)` | 按类型权重与复杂度分配，总和等于预算（±1 bit） |
| `reorder_bframes(types)` | 显示序 → 编码序，GOP 内为 `I P B...` |
| `budget_check(bits, budget)` | 返回 `{"total", "diff", "ok"}`，容差 1 bit |
| `validate_plan(types, bits, budget, keyint)` | 检查 I 间隔、总比特、类型与 B 的参考合法性 |
| `plan_batch(clips, config)` | 批量独立规划各片 |

每个函数的完整契约见 `lookahead/plan.py` 的 docstring。
