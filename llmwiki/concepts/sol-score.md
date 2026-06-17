---
type: concept
title: SOL Score — 度量的数学与设计
updated: 2026-06-17
sources: [arXiv:2603.19173 §4.3/§5.2, src/sol_execbench/sol_score.py]
tags: [metric, sol-score, scoring]
---

# SOL Score：度量的数学与设计

SOL-ExecBench 的核心度量。把"候选 kernel 填补了多少 baseline→硬件极限的优化空间"映射到一个有界、可解释、可跨问题平均的分数。

## 1. 公式

报告 §4.3 Eq.(2)，与代码 `src/sol_execbench/sol_score.py::sol_score` 完全一致：

```
                 1
S(T_k) = ────────────────────────
          1 + (T_k − T_SOL)
              ─────────────
              (T_b  − T_SOL)
```

等价形式 Eq.(3)：`S(T_k) = (T_b−T_SOL) / [(T_k−T_SOL) + (T_b−T_SOL)]`。

符号（见 [[glossary]]）：
- `T_k` = 候选 kernel 实测运行时间（[[concepts/timing-and-reproducibility]] 测得）
- `T_b` = scoring baseline 运行时间（高性能实现，**0.5 中点锚**）
- `T_SOL` = SOLAR 估的硬件光速（[[concepts/speed-of-light-and-solar]] 推导）

分母里的 `T_b − T_SOL` 就是"**优化头**（headroom）"——基线离硬件极限还有多远。SOL Score 衡量的是候选**吃掉了多大比例的这段头**。

## 2. 代码实现（含边界处理）

```python
# src/sol_execbench/sol_score.py
def sol_score(t_k, t_b, t_sol):
    denom_gap = t_b - t_sol
    if denom_gap <= 0:
        return 1.0 if t_k <= t_sol else 0.0
    return 1.0 / (1.0 + (t_k - t_sol) / denom_gap)
```

注意那个 `denom_gap <= 0` 分支：当 baseline 已经 ≤ SOL（头被吃光、问题已解），公式会退化/除零。代码的处理是——候选达到 SOL 给 1.0，否则 0.0。报告 §4.3 对应的工程决定是：**当 `T_b → T_SOL` 时认为该问题已解决，不再为新提交评分**。

## 3. 三个锚点性质（报告 §4.3）

| 条件 | 分数 | 含义 |
|------|------|------|
| `T_k = T_b` | **S = 0.5** | 持平 baseline |
| `T_k = T_SOL` | **S = 1.0** | 触到硬件极限（满分） |
| `T_k → ∞` | **S → 0** | 越慢越趋近 0，平滑衰减 |

于是 [0,1] 被自然切成三段语义区间：
- `S < 0.5`：**低于** baseline；
- `0.5 < S < 1`：**超过** baseline 但未到 SOL；
- `S = 1`：SOL 级。

> **为什么 baseline 拿 0.5 而不是 0？** 报告 §4.3 解释：让 baseline 落在中点，度量才能在一条有界刻度上同时区分"低于/高于/极限"三种状态；低于基线区间里分数随变慢平滑趋零。报告说也试过 clip/sigmoid 等变体来达到同样目的，最终选了这个公式因为**最简单**。

## 4. 非线性：越靠近 SOL，同样的提速越值钱

公式是 `T_k` 的凸函数。报告 §4.3 图 5 强调它的**非线性**：同样大小的运行时间改善，发生在越靠近 SOL 的区域，带来的分数增益越大。这把激励对齐到了正确方向——**鼓励去啃最后那段最难的、最接近硬件极限的优化**，而不是在远离极限处刷容易的提速。

## 5. 正确性闸门（Correctness gate）

报告 §4.3：引入正确性指示 `C ∈ {0,1}`。**未通过验证的 kernel `C=0`，无论多快，性能分一律为零。** 在代码里这体现为：只有走到 `EvaluationStatus.PASSED` 的 workload 才带 performance；任何 INCORRECT/REWARD_HACK/ERROR 状态都不产出 performance（见 [[components/data-models]] 的 nullable 表与 [[components/evaluation-driver]] 的状态机）。

**先证明算对，再谈多快**——这是把性能度量建立在可信之上的硬约束。

## 6. 套件级聚合：算术平均

报告 §4.3 Eq.(4)：N 个问题的总分 = 各问题分的算术平均

```
S̄ = (1/N) Σ_j  C_j · S_j
```

为什么用**算术平均**而不是几何平均之类：因为每个 `S_j` 已经被归一化到 [0,1] 且**跨问题语义一致**（都表示"填补了多大比例的头"），所以直接平均能在套件层面保持同一解释。报告还提到这能自然扩展到 agentic 系统的 best-of-k 设定（每问题多候选取最好）。

## 7. 为什么不用 speedup：关键实验（报告 §5.2）

这是支撑整套范式转变的实证（见 [[concepts/design-philosophy]]）：

- **speedup 与"离 SOL 多远"几乎不相关**：在 log-log 尺度上 Pearson **r=0.10**。一个 kernel 可以比 PyTorch 快 10×，却仍离硬件 SOL 远 >10×。speedup-only 的尺子会把这种 kernel 排得很靠前，掩盖巨大头部空间。
- **SOL Score 几乎完美追踪"填补的头部比例"**：与 *fraction of headroom reclaimed* = `(T_ref−T_k)/(T_ref−T_SOL)` 的 Pearson **r=0.981**。
- 同样的 speedup 会因 SOL 距离不同而映射到**很不一样**的 SOL Score——证明 S 不是 speedup 的重新标签，而是把"提速幅度"和"离极限多近"两个轴压进一个有界值。

验证实验里，agentic optimizer 产出的 baseline 中位 SOL Score = **0.732**，明显高于 0.5 中点又留有清晰头部空间。

## 8. 审计信号（Audit signals）

公式假设 `T_b > T_SOL` 且 `T_k ≥ T_SOL`。一旦现实中被违反（例如候选比理论极限还快 `T_k < T_SOL`），报告 §4.3 的处理是：**当作审计信号上报**，触发两类复查——
1. SOLAR 的 bound 是不是算松了（见 [[concepts/speed-of-light-and-solar]] 的局限）；
2. 是不是 reward hacking（见 [[concepts/reward-hacking-defenses]]）。

## 延伸阅读
- 输入 `T_SOL` 怎么来：[[concepts/speed-of-light-and-solar]]
- `T_k` 怎么测准测稳：[[concepts/timing-and-reproducibility]]
- `C`（正确性）怎么判：[[concepts/correctness-protocol]]
- 为什么换这把尺子：[[concepts/design-philosophy]]
