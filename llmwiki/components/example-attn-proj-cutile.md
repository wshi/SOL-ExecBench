---
type: component
title: 样例剖析 — attn_proj_residual_cutile（cuTile 实现）
updated: 2026-06-22
sources: [examples/cutile/jamba_attn_proj/{kernel.py,solution_cutile.json,definition.json}, examples/cute_dsl/jamba_attn_proj/kernel.py]
tags: [example, cutile, gemm, blackwell, dsl, solution]
---

# 样例剖析：`attn_proj_residual_cutile`

仓库里**唯一**的 cuTile 样例算子（DSL 清单见 [[components/data-models]] §3.1）。本页逐段剖析它的 cuTile 实现：算什么、怎么用 cuTile 的 tile-level 编程模型写出来、与 [[components/data-models]] 的契约怎么对接、以及和同问题的 `cute_dsl` 版本有何本质差异。

> 证据锚点：`examples/cutile/jamba_attn_proj/kernel.py`（源码即下文行号引用）。

---

## 1. 问题契约（算什么）

来自 Jamba-Reasoning-3B 的**注意力输出投影 + 残差**，定义 `ai21labs_..._attention_output_projection_with_residual`：

```python
# reference（PyTorch ground truth）
projected = torch.matmul(attn_output, o_proj_weight.t())   # (M,K) @ (K,N) -> (M,N)
output    = projected + residual
```

- 轴：`hidden_size` = **const 2560**；`batch_size` / `seq_len` = var（workload 绑定）。
- 张量全 **bf16**：`attn_output (B,S,H)`、`residual (B,S,H)`、`o_proj_weight (H,H)`、`output (B,S,H)`。
- 形状特点：H 是方阵投影，所以 **K = N = 2560**，M = batch×seq。workload 给的三个形状 M ∈ {512, 2048, 586}——注意 **586 不是任何 tile 的整数倍**，对齐处理是实现必须答的题（见 §3 padding）。

`DPS=false`（return-value 风格）：`run` 直接 `return output`，不写预分配 buffer——与 [[components/data-models]] §3.1 观察一致（样例普遍用 return-value）。

---

## 2. `run` 的整体结构（host 侧编排）

`kernel.py:53-78` 的 `run` 把任务拆成**两个独立 cuTile kernel**，串在当前 CUDA stream 上：

```python
M = attn_output.numel() // H;  K, N = H, H
a_flat   = attn_output.contiguous().view(M, K)          # (M,K)
res_flat = residual.contiguous().view(M, N)             # (M,N)

# ① GEMM   (tm, tn, tk = 128, 256, 64)
projected = torch.empty((M, N), ...)                    # 中间张量，物化到 global memory
ct.launch(stream, (grid_x*grid_y, 1, 1), matmul_kernel,
          (a_flat, o_proj_weight.t().contiguous(), projected, tm, tn, tk))   # 位置参数

# ② Residual add  (add_tm, add_tn = 32, 256)
output = torch.empty_like(projected)
ct.launch(stream, (ceil(M/add_tm), ceil(N/add_tn), 1), add_kernel,
          (projected, res_flat, output, add_tm, add_tn))
return output.view(shape)
```

几个 host 侧要点：

- **权重转置在 host 完成**：传给 kernel 的是 `o_proj_weight.t().contiguous()`，即把 `(N,K)` 权重显式转成行主序 `(K,N)`，于是 device kernel 只需做标准 `A(M,K)·B(K,N)`，无需在 kernel 里处理转置布局。代价是一次 host 侧 `contiguous()` 拷贝。
- **grid 编码方式不同**：matmul 把二维 tile 网格**压平成一维** `(grid_x*grid_y,)`，在 kernel 内用 `swizzle_2d` 还原 `(bid_m,bid_n)`（为 L2 复用做分组重排，见 §3）；add 直接用**二维 grid** `(ceil(M/32), ceil(N/256))`（host 侧用 `math.ceil`）。
- **两次 launch 都显式传 `torch.cuda.current_stream()`**——和评测器的计时/流约定对齐（见 [[concepts/timing-and-reproducibility]]、[[concepts/reward-hacking-defenses]] 对 stream 注入的检查）。

---

## 3. `matmul_kernel` 逐段（cuTile 的核心）

`kernel.py:25-40`：

```python
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    GROUP_SIZE_M = 8
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)     # ← 一维 bid 还原为 (m,n) tile 坐标
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)    # ← 累加器恒为 fp32
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=ZERO).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=ZERO).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)             # ← tile 级 MMA，累加进 fp32
    accumulator = ct.astype(accumulator, C.dtype)           # ← 末尾才降回 bf16
    ct.store(C, index=(bidx, bidy), tile=accumulator)
```

要点（这正是 cuTile "tile 即一等公民"的体现）：

1. **tile-level 而非 thread-level**：`ct.load/ct.mma/ct.store` 直接以 **128×256 / 128×64 的 tile** 为操作单位——程序员不写 thread index、不手搬 shared memory，由编译器把一个 CTA（或 §3.5 的 CTA pair）映射到 tile 上。这是 cuTile 与 CUDA C++ 最大的抽象差异。
2. **`swizzle_2d` 分组重排（L2 复用）**：`kernel.py:8-22` 是经典 Triton 风格的 group-M swizzle——把相邻的 program id 按 `GROUP_SIZE_M=8` 行成组遍历，让同时在跑的 CTA 更可能复用 L2 里的同一块 A/B，减少 DRAM 读。`group_size_m=min(num_bid_m-first_bid_m, GROUP_SIZE_M)` 处理最后一组不满的边界。
3. **fp32 累加、bf16 进出**：累加器恒 `float32`，循环里对 bf16 操作数做 MMA 累加进 fp32，**最后一步**才 `astype` 回 `C.dtype`(bf16)。这把舍入误差只发生一次，对 [[concepts/correctness-protocol]] 的 bf16 容差（atol/rtol 1e-2）友好。（`tfloat32` 分支只在输入是 fp32 时触发；本问题 bf16 不走它。）
4. **`padding_mode=ZERO` 解决非对齐**：M=586、K=N=2560 对 tile (128,256,64) 都不整除。越界 load 补 0——对 GEMM 累加是中性的（加 0 不改结果），store 按 tile 坐标写、越界部分由框架裁掉。这是它能正确跑 M=586 那条 workload 的关键。
5. **`num_ctas=ct.ByTarget(sm_100=2)` —— Blackwell 专属**：在 sm_100（B200）上每个 tile 用 **2 个 CTA**（CTA pair / cluster）协作完成一个 MMA tile，对应 Blackwell 第 5 代 Tensor Core 的双-CTA MMA 能力。**这是该样例标 `@requires_cutile`（需 SM 100+）的根因**（见 [[components/data-models]] §3.1 与 [[concepts/timing-and-reproducibility]] 的 Blackwell 门槛）。

---

## 4. `add_kernel`（残差加）

`kernel.py:43-50`，平凡的逐 tile 二元加：

```python
@ct.kernel
def add_kernel(A, B, C, tm, tn):
    bidx, bidy = ct.bid(0), ct.bid(1)                       # 直接用二维 grid
    a = ct.load(A, index=(bidx,bidy), shape=(tm,tn), padding_mode=ZERO)
    b = ct.load(B, index=(bidx,bidy), shape=(tm,tn), padding_mode=ZERO)
    ct.store(C, index=(bidx,bidy), tile=(a + b))            # 张量级 a+b，无显式标量循环
```

tile 32×256，二维 grid。同样靠 `padding_mode=ZERO` 兜非对齐。注意 `a+b` 是**整块 tile 的张量加**，不是逐元素标量循环——再次体现 cuTile 把 tile 当值来运算。

---

## 5. 一个反直觉的关键事实：**它并没有"融合"**

定义把这题叫 **fused** o_proj + residual，reference docstring 还专门描述了理想融合：

> *"compute matmul tiles in registers, add residual directly to register-held results, write final result to global memory (eliminating intermediate write)"*

但 cuTile 这版实现**没有做这个融合**：它是「GEMM kernel → 把 `projected` 物化到 global memory → 再启动 add kernel 读回来加 residual」。即多了一趟 (M×N) bf16 中间张量的**写出 + 读回**往返。

- 这不影响**正确性**（结果与 reference 一致），但留下了明显的**性能空间**：真正的融合应在 matmul 的 epilogue 里直接把 residual 加进 fp32 累加器再写出，省掉中间张量。
- 因此它更像一个**能跑、能对、展示 cuTile GEMM 写法**的 baseline 解，而非冲 SOL Score 的优化解。按 [[concepts/sol-score]]，这种"对但未极致优化"的解会落在 0.5（baseline）附近甚至以下。

> 诚实标注（[[CLAUDE]] §3.6）：以上是静态代码阅读得出的结构性判断；没有实测 latency，"性能空间"是定性推断而非测量结论。

---

## 6. 与 `cute_dsl` 版本对照（同一问题，两种 DSL）

同目录问题在 `examples/cute_dsl/jamba_attn_proj/kernel.py` 还有一版 CuTe DSL 实现。两者**分工不同**，对照很有信息量：

| 维度 | `cutile` 版 | `cute_dsl` 版 |
|---|---|---|
| GEMM | **手写 cuTile `matmul_kernel`**（tile MMA + swizzle） | **直接 `torch.matmul`**（不手写） |
| 残差加 | 手写 cuTile `add_kernel` | 手写 CuTe DSL `_add_kernel` |
| 抽象层次 | tile 级：`ct.load/mma/store`，无 thread/shared 细节 | thread-value 级：`thread_idx`、`copy_atom`、`make_tiled_copy_tv`、`partition_S`、显式谓词 `frgPred` |
| 编译/缓存 | 每次 `ct.launch` 即时派发 | `cute.compile` 一次后 `_compiled_add` 全局缓存复用 |
| 非对齐处理 | `padding_mode=ZERO` | 显式 identity tensor + `elem_less` 谓词 mask |
| 融合 | 否（两 kernel + 中间张量） | 否（GEMM 在 torch，add 单独 kernel） |

要点：

- **cuTile 版是更"完整"的手写实现**——它把 GEMM 也用 DSL 写了；cute_dsl 版只手写了残差加、GEMM 外包给 torch。所以想看"cuTile 怎么写一个带 swizzle 的 tiled GEMM"，本页是仓库里唯一样例。
- 抽象层次差一档：cuTile 隐藏了 thread/shared/copy-atom，CuTe DSL 把这些都暴露给程序员手控。两者都映射到同一硬件，但 cuTile 代码量与心智负担明显更低。
- 两版**都没真正融合**——印证 §5 不是 cuTile 的局限，而是这两个样例解共同的"留白"。

---

## 7. cuTile 编程模型速记（从本例提炼）

| cuTile 原语 | 作用 | 本例用法 |
|---|---|---|
| `@ct.kernel` / `@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))` | 声明 tile-level kernel；`num_ctas` 按目标架构指定每 tile CTA 数 | matmul 用 2-CTA(Blackwell)，add 用默认 |
| `ct.Constant[int]` | 编译期常量参数（tile 尺寸） | `tm/tn/tk` |
| `ct.bid(axis)` | 取 block id | grid 坐标 |
| `ct.cdiv` / `ct.num_tiles` | tile 网格尺寸 / 某轴 tile 数 | swizzle、K 循环 |
| `ct.full(shape, v, dtype)` | 建片上 tile（含累加器） | fp32 accumulator |
| `ct.load(T, index, shape, padding_mode)` | 按 tile 坐标载入一块，自动边界填充 | A/B/residual 载入 |
| `ct.mma(a, b, acc)` | tile 级矩阵乘累加（→Tensor Core） | GEMM 主循环 |
| `ct.astype` / `.astype` | tile dtype 转换 | bf16↔fp32、tf32 |
| `ct.store(T, index, tile)` | 按 tile 坐标写回，自动裁边界 | 写 C / output |
| `ct.launch(stream, grid, kernel, args)` | host 侧启动 | run 里两次 |

---

## 延伸阅读
- DSL 总清单与 DPS 观察：[[components/data-models]] §3 / §3.1
- 为何需要 SM 100+ / Blackwell：[[concepts/timing-and-reproducibility]]
- 这种"对但未优化"解的得分定位：[[concepts/sol-score]]
- bf16 容差如何判正确：[[concepts/correctness-protocol]]
- stream 显式传入与反作弊：[[concepts/reward-hacking-defenses]]
