# llama.cpp KV Cache 与"分页/连续缓存"源码笔记

- 仓库：`E:\llama.cpp`（origin = https://github.com/ggml-org/llama.cpp.git）
- 版本：`555881ebc8b0fc0402b30e09258a32a7bfd13c52`（2026-07-24），`llama-server version: 10121 (555881ebc)`
- 阅读方式：ripgrep 关键词检索 + 逐段精读；下文所有行号均来自本仓库的**实际源码**，可直接跳转核对。
- 阅读时的仓库状态：`master...origin/master`，工作区干净（`git status -sb` 无改动）。

---

## 0. 一句话结论（先看这里）

1. **llama.cpp 里没有 PagedAttention**。全仓库（`*.cpp/*.h/*.cu/*.cuh/*.md`）检索 `paged|PagedAttention|paged_attention|block_table|block_tables`，命中的**只有 `vendor/miniaudio/miniaudio.h`**（音频环形缓冲，与注意力无关）。llama.cpp **不存在**名为 PagedAttention 的实现，也不存在 vLLM 意义上的 block table。
2. llama.cpp 的 KV Cache 是**一整块连续的、按层切分的 tensor**，形状 `[n_embd_k_gqa, kv_size, n_stream]`（K）和 `[n_embd_v_gqa, kv_size, n_stream]`（V），见 `src/llama-kv-cache.cpp:231-232`。
3. "分页"这件事在 llama.cpp 里被替换为两层机制：
   - **cell 元数据层**（`llama_kv_cells`）：每个 cell 记录 `pos` / `shift` / `seq` bitset，实现"哪些 token 属于哪些序列"以及可回收性判断；
   - **环形缓冲 + 线性扫描分配器**（`find_slot()`，`src/llama-kv-cache.cpp:894`）：用一个移动的 `v_heads[stream]` 游标在固定大小的 cell 数组里找空位。
4. **没有动态申请/释放**：KV 显存（或内存）在 context 创建时**一次性分配完毕**，运行时只做 cell 的复用（覆盖写），不存在运行时向驱动申请新块的行为。所以 llama.cpp 不会"碎片化增长"，它的瓶颈是**一开始能不能分配下**（见下方 OOM 实测）。
5. `--kv-unified` 控制的是 `n_stream`：统一缓存 = 1 个共享 stream，非统一 = `n_seq_max` 个独立 stream（`src/llama-kv-cache.cpp:82`）。**总 cell 数不变**（都等于 `n_ctx`），变的是"每个序列能用多少"。

---

## 1. 检索命令与命中情况

```bash
rg -n "llama_kv_cache|kv_cache" src/ tools/server/
rg -n "paged|PagedAttention|paged_attention|block_table|block_tables" --glob '*.{cpp,h,cu,cuh,md}'
rg -n "find_slot|seq_rm|seq_cp|apply_ubatch|prepare\(" src/llama-kv-cache.cpp
rg -n "kv_unified|n_ctx_seq|n_seq_max" src/ common/ tools/server/
rg -n "n_kv|nbatch_fa|k_VKQ" ggml/src/ggml-cuda/fattn-tile.cuh
```

实际文件（与本任务预期一致）：

| 任务预期文件 | 实际情况 |
|---|---|
| `src/llama-kv-cache.cpp` | 存在，2641 行 |
| `src/llama-kv-cache.h` | 存在，434 行 |
| `src/llama-context.cpp` | 存在 |
| `src/llama-graph.cpp` | 存在 |
| `ggml/src/ggml-cuda/` fattn | 存在：`fattn.cu`、`fattn-common.cuh`、`fattn-tile.cu(h)`、`fattn-vec.cuh`、`fattn-mma-f16.cuh` |
| （额外发现） | `src/llama-kv-cells.h`、`src/llama-kv-cache-iswa.{cpp,h}`（SWA 混合）、`src/llama-kv-cache-dsa.{cpp,h}`、`src/llama-kv-cache-dsv4.{cpp,h}`（DeepSeek 稀疏/压缩注意力专用缓存） |

---

## 2. 核心数据结构

### 2.1 `llama_kv_cells` —— 每个 cell 的元数据（`src/llama-kv-cells.h:32`）

这是理解 llama.cpp KV 管理的关键。**一个 cell = 一个 token 位置在一个 stream 里的槽位**。

私有字段（`src/llama-kv-cells.h:458-499`）：

| 字段 | 行号 | 含义 |
|---|---|---|
| `std::set<uint32_t> used` | `:462` | 已使用 cell 的下标集合，`used_min()`/`used_max_p1()` 靠它做 O(1)/O(log n) |
| `std::vector<llama_pos> pos` | `:464` | 每 cell 的序列位置；`pos[i] == -1` 表示空 cell |
| `std::vector<llama_kv_cell_ext> ext` | `:467` | M-RoPE 等二维位置（`x`,`y`），见 `:13-28` |
| `std::vector<llama_pos> shift` | `:484` | 自上次 `reset_shift()` 以来累计的位置偏移，用于 K-shift 批量化 |
| `std::vector<seq_set_t> seq` | `:486-489` | **`std::bitset<LLAMA_MAX_SEQ>`**，标记该 cell 被哪些序列共享 |
| `std::map<llama_pos,int> seq_pos[LLAMA_MAX_SEQ]` | `:499` | 每个序列的位置多重集，用于快速取 `seq_pos_min/max` |

`LLAMA_MAX_SEQ = 256`（`src/llama-cparams.h:8`）。

关键操作：

- `seq_rm(i, seq_id)`（`:238`）：从 cell 里摘掉一个序列的位；**只有当 bitset 变空时才真正释放 cell**（`pos[i] = -1`, `used.erase(i)`），返回 `true`。
- `seq_add(i, seq_id)`（`:309`）：把一个序列加到已有 cell 上 —— 这就是**序列间共享前缀**的实现方式，**不复制 K/V 数据**。
- `seq_keep(i, seq_id)`（`:261`）：只保留一个序列，其余位全部清掉。
- `rm(i)`（`:222`）：无条件清空 cell。
- `pos_set(i,p)`（`:395`）/ `pos_add(i,d)`（`:413`）/ `pos_div(i,d)`（`:442`）：位置写入与平移。

> 参考计数语义：cell 的 `seq` 是 bitset，天然支持"一个 cell 被 N 个序列引用"。释放是引用计数式的。但因为**不同序列共享同一块物理 K/V**，所以不存在写时复制（COW）——写这个 cell 会同时影响所有引用它的序列。

### 2.2 `llama_kv_cache` —— 缓存主体（`src/llama-kv-cache.h:20`）

继承 `llama_memory_i`。

关键成员（`src/llama-kv-cache.h`）：

| 成员 | 行号 | 说明 |
|---|---|---|
| `const uint32_t n_seq_max` | `:238` | 最大序列（并行 slot）数 |
| `const uint32_t n_stream` | `:239` | `unified ? 1 : n_seq_max`（实参见 `:82`），**`GGML_ASSERT(n_stream == 1 \|\| n_stream == n_seq_max)` at `:133`** |
| `const uint32_t n_pad` | `:242` | KV 尺寸对齐粒度；server 传 1（`src/llama-model.cpp:2140`），单序列场景有效对齐在 `get_n_kv()` 里提升到 256 |
| `std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs` | `:266` | KV 的 ggml context 与后端 buffer，**一次性分配** |
| `std::vector<uint32_t> v_heads` | `:270` | 每个 stream 的搜索起点（环形游标），**注：它是加速用的，不属于 KV state** |
| `std::shared_ptr<llama_kv_cells_vec> v_cells_impl` / `llama_kv_cells_vec & v_cells` | `:275-277` | `std::vector<llama_kv_cells>`，每个 stream 一个 |
| `std::vector<uint32_t> seq_to_stream` | `:280` | seq_id -> stream_id 映射 |
| `std::vector<kv_layer> layers` | `:285` | 参与缓存的层 |
| `std::unordered_map<int32_t,int32_t> map_layer_ids` | `:288` | 模型层号 -> 缓存层号 |
| `stream_copy_info sc_info` | `:283` | 待执行的跨 stream 拷贝（延迟到 `update()`） |

`kv_layer`（`src/llama-kv-cache.h:224-234`）：每层保存整个 3D tensor `k`/`v`，以及 `n_stream` 个 2D view `k_stream[s]`/`v_stream[s]`。

### 2.3 `slot_info` —— 一次 ubatch 的落位方案（`src/llama-kv-cache.h:34-92`）

```
struct slot_info {
    uint32_t s0, s1;                    // 涉及的 stream 区间 [s0, s1]
    std::vector<llama_seq_id> strm;     // [ns] 每个子序列去哪个 stream
    std::vector<idx_vec_t>    idxs;     // [ns][n_tokens] 每个 token 落到哪个 cell
};
```

- `head()`（`:45`）返回 `idxs[0][0]`。
- `is_contiguous()`（`:77`）：检查 `idxs[0]` 是否是从 `head()` 开始的连续递增序列 —— 这正是"连续槽位"的判定，用于判断能否走连续写路径。

`llama_kv_cache_context`（`:323`）是 `llama_memory_i` 的运行时上下文，持有 `sinfos` / `ubatches` / `n_kv`（`:421-433`），执行 `next()`/`apply()`（`:355-356`）。

---

## 3. 内存分配：一次性、连续、按层 3D

构造函数 `llama_kv_cache::llama_kv_cache(...)`（`src/llama-kv-cache.cpp:64`）：

```cpp
// :82
n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), ...
```

- `:140-143` 为每个 stream 建一个 `llama_kv_cells`，容量 `kv_size`。
- `:146-153` 建立 `seq_to_stream`：**unified 时所有 seq 映射到 stream 0；非 unified 时 `seq_to_stream[s] = s`**。
- `:163-248` 逐层创建 tensor：

```cpp
// :231-232
ggml_tensor * k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
ggml_tensor * v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);
// :240-243 每 stream 切一个 2D view
k_stream.push_back(ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]));
```

- `:274-293` 按 buffer type 分组分配（`ggml_backend_alloc_ctx_tensors_from_buft`），分配失败直接 `throw std::runtime_error("failed to allocate buffer for kv cache")`（`:286`）——**这就是 8 GiB 显卡上 32k 上下文失败的抛出点**。
- `:289` 打印 `KV buffer size = %8.2f MiB`
- `:299-302` 打印总量：
  `size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (type): ... MiB, V (type): ... MiB`
  其中 `(n_seq_max/n_stream)` 直接暴露了是否 unified。

**总 KV 字节数**（`size_k_bytes()`/`size_v_bytes()`，`:1810`/`:1820`）≈
`n_layer × n_stream × kv_size × (n_embd_k_gqa × sizeof(type_k) + n_embd_v_gqa × sizeof(type_v))`。

**没有一个字节是运行时按需分配的** —— 这一点决定了 llama.cpp 与 vLLM 在显存管理哲学上的根本差异。

---

## 4. slot 搜索：环形缓冲 + 线性扫描（`find_slot`）

`llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont)`（`src/llama-kv-cache.cpp:894`）：

1. `:962-970` 非 unified 时把 ubatch 按序列拆开：`n_seqs = ubatch.n_seqs_unq; n_tokens /= n_seqs;`
2. `:995-997` 取当前 stream 的 cell 数组和游标 `head_cur = v_heads[stream]`。
3. `:999-1003` 启发式：如果 head 之前有足够空闲，直接把 `head_cur` 归零以填满前部。
4. `:1005-1008` `n_tokens > cells.size()` 直接报错返回空 —— 这就是"单请求超过 slot 上下文"时的失败点。
5. `:1014` `const uint32_t n_test = cont ? n_tokens : 1;` —— 连续模式要求**一次性找到连续 n_tokens 个可用 cell**；非连续模式逐 token 试。
6. `:1017-1021` 到尾部就环绕回 0（**环形缓冲**）。
7. `:1038` 可用的判定：
   ```cpp
   bool can_use = cells.is_empty(idx);
   if (!can_use && cells.seq_count(idx) == 1) {
       // SWA 淘汰：位置超出滑窗的旧 cell 可被覆盖
       if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1))
           can_use = true;
   }
   ```
   **注意 `:1043-1047` 被显式注释掉的因果复用分支**：曾经允许"同序列的 future token 被覆盖"，现在被禁用了（注释说明"最好提前 purge 掉 future token"）。
8. `:1068-1080` 凑齐 `n_tokens` 就返回；扫完一圈仍不够就 `return {}`（**分配失败**，上层 `prepare()` 返回空 -> `LLAMA_MEMORY_STATUS_FAILED_PREPARE`）。

复杂度：最坏 O(kv_size) 线性扫描，`v_heads` 只是把起点往前推的优化，**没有任何哈希/块表**。

### 4.1 失败回滚与批处理

- `prepare()`（`:747`）：对每个 ubatch 先 `find_slot`（`:765`），再记录旧 cell 状态（`:774-785`），再 `apply_ubatch`（`:788`）。**最后把 cell 状态与 `v_heads` 全部回滚**（`:793-804`），只留下"方案"。真正的写入发生在 `llama_kv_cache_context::apply()`。
- 这样设计的原因：一个 batch 被切成多个 ubatch，任何一个 ubatch 找不到位置，整个 batch 必须原子性失败。

### 4.2 落位 `apply_ubatch`（`:1093`）

- `:1124` 若目标 cell 非空，先 `cells.rm(idx)` 并记录被覆盖的最大位置 `seq_pos_max_rm`。
- `:1127` `cells.pos_set(idx, ubatch.pos[i])`
- `:1130-1135` 二维位置（M-RoPE）写入 `ext`
- `:1137-1139` 对 batch 里该 token 的每个 seq_id 调 `cells.seq_add` —— **多序列共享同一 cell 在这里发生**
- `:1146-1161` 维护不变量"序列的 `[pos_min, pos_max]` 区间内位置齐全"：把小于被覆盖位置的部分 purge 掉
- `:1163-1168` `head = idxs.back() + 1`，把环形游标推到本次槽位之后

---

## 5. 序列操作：`seq_rm` / `seq_cp` / `seq_add` / `seq_div`

| API | 行号 | 语义 |
|---|---|---|
| `seq_rm(seq_id, p0, p1)` | `:379` | 删除区间内的 cell。`seq_id >= 0` 走 bitset 摘除（`:406`，`seq_rm` 返回 true 才真正释放）；`seq_id == -1` 走"匹配任意序列"的全清分支（`:417-442`）。**释放后会尝试把 `head` 拉回到新空出的最小下标**（`:413-416`）。 |
| `seq_cp(src, dst, p0, p1)` | `:447` | 见下 |
| `seq_keep(seq_id)` | `:539` | 只留一个序列 |
| `seq_add(seq_id, p0,p1, shift)` | `:566` | 位置平移 |
| `seq_div(seq_id, p0,p1, d)` | `:616` | 位置整除（RoPE 上下文压缩类用法） |

**`seq_cp` 的两种路径（重要）**：

- **同一 stream**（`s0 == s1`，`:459-488`）：**不复制任何数据**，只是对区间内每个 cell 调 `cells.seq_add(i, seq_id_dst)`（`:483`）。这就是"两个序列共享同一段 KV"——即 **prefix sharing / 系统提示共享** 的实现。
- **跨 stream**（`:490-532`）：`GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")`（`:502`）—— 只支持整块拷贝。拷贝被**延迟**到 `update()`（`:505-506` push 进 `sc_info`），届时用 `ggml_backend_tensor_copy` 逐层做 `k_stream[ssrc] -> k_stream[sdst]`（`:841-849`）。`update()` 里还有 `assert(n_stream > 1 && "stream copy should never happen with a single stream")`（`:824`）—— **unified 模式下不存在跨 stream 拷贝**。

---

## 6. K-shift（上下文滑动/压缩）

`update(lctx, do_shift, sc_info)`（`:813`）在 `do_shift` 时：

- `:853-855` 若模型不支持则 `GGML_ABORT`（`get_can_shift()`，`:1171`）
- `:861-882` 构建 `build_graph_shift()` 计算图，对 K 做 RoPE 反向旋转到新位置
- `:884-888` `cells.reset_shift()`

`get_can_shift()`（`:1171-1180`）对 `LLM_ARCH_STEP35` 与 `n_pos_per_embd() > 1`（M-RoPE）返回 false。

---

## 7. "分页 / 统一缓存 / 连续缓存"到底指什么

### 7.1 llama.cpp 的"分页"= cell 槽位复用，不是虚拟内存分页

- 物理承载：**每个 stream 一整块连续 tensor**（`:231-232`）。
- 逻辑承载：`llama_kv_cells` 的 `pos[]` 数组，`pos[i] == -1` 即空闲。
- "分配"= 在固定数组里找一个 `pos == -1` 的下标并写入；"释放"= 把 `pos` 置 -1。
- **没有 page、没有 block table、没有物理块与逻辑块的二级映射、没有按需申请**。

### 7.2 "连续"（contiguous）

两种含义，别混淆：

1. **连续的物理缓冲**：每个 stream 的 K/V 是一整块连续显存。
2. **连续的槽位**：`slot_info::is_contiguous()`（`llama-kv-cache.h:77`）判定一次 ubatch 的所有 token 是否落在 `[head, head+n)` 这段连续 cell 上；`find_slot(ubatch, /*cont=*/true)` 会**要求**连续。当前 `init_batch()` 走的是 `find_slot(ubatch, false)`（`llama-kv-cache.cpp:765`），即允许非连续落位。

### 7.3 "统一缓存"（unified KV cache）

来自 `src/llama-context.cpp:286-300`：

```cpp
if (cparams.kv_unified) {
    cparams.n_ctx_seq = cparams.n_ctx;              // 每个序列可用满 n_ctx
} else {
    cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
    cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);
    if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
        cparams.n_ctx = cparams.n_ctx_seq * cparams.n_seq_max;   // 日志写 "rounding down"
    }
}
```

而 `cparams.n_ctx_seq` 就是传给缓存的 `kv_size`（`src/llama-model.cpp:2138`：`/* attn_kv_size */ cparams.n_ctx_seq`）。

于是：

| | `n_stream` | `kv_size` | 总 cell 数 | 单个序列上限 |
|---|---|---|---|---|
| **非 unified**（默认，显式给 `-np N`） | `N` | `pad(n_ctx/N, 256)` | `N × kv_size` ≈ `n_ctx` | `n_ctx/N` |
| **unified** | `1` | `n_ctx` | `n_ctx` | `n_ctx` |

**总显存占用基本相同**（都 ≈ `n_ctx` 个 cell），差别是**分配策略**：非 unified 是**静态等分**（每个 slot 独占一段，互相借不到），unified 是**共享池**（谁能用谁用，闲置的别人可以占）。

### 7.4 连续批处理（continuous batching）

`--cont-batching`（默认开启）让 server 在 `update_slots()` 里把多个 slot 当前要算的 token 合并进同一个 batch。它对 KV 的影响是**间接但关键**的：不同序列的 token 会被塞进同一个 ubatch，因此

- 非 unified 下必须在 `find_slot` 前把 ubatch 按序列切开（`llama-kv-cache.cpp:965-970`），且 `init_batch` 用 `balloc.split_equal(...)` 而不是 `split_simple`（`:709`）；
- 图构建时 `kq_mask` 被组织成 4D `[n_kv, n_tokens/n_stream, 1, n_stream]`（`src/llama-graph.cpp:33-38`，`n_stream = cparams.kv_unified ? 1 : ubatch.n_seqs_unq`）。

也就是说，**非 unified + 连续批处理会让 attention 图带一个额外的 stream 维度**，每个 stream 有自己独立的 n_kv 和 mask。

---

## 8. llama-server 参数对 KV Cache 的实际影响

### 8.1 `-np / --parallel N`

- 默认 `n_parallel = -1`（auto，`common/arg.cpp:1288`）。
- **`n_parallel < 0` 时 server 强制 `n_parallel = 4` 且 `kv_unified = true`**（`tools/server/server.cpp:146-151`）：
  > `n_parallel is set to auto, using n_parallel = 4 and kv_unified = true`
- `--kv-unified` 的默认值是 **false**（`common/common.h:574`：`bool kv_unified = false;`），由 `-kvu/--kv-unified`、`-no-kvu/--no-kv-unified` 覆盖（`common/arg.cpp:1598-1602`）。
- 因此：**显式写 `-np 10` 而不写 `--kv-unified`，得到的是非 unified 缓存**（10 个独立 stream，每个 `n_ctx/10`）。这是很容易踩的坑。
- server 侧把每个 slot 的 `n_ctx` 设为 `llama_n_ctx_seq(ctx)`（`tools/server/server-context.cpp:1252`、`:1306`），并打印：
  > `initializing, n_slots = %d, n_ctx_slot = %d, kv_unified = '%s'`（`:1270-1271`）

### 8.2 `--kv-unified`

- 影响 `n_stream`（1 vs `n_seq_max`），进而影响：
  - 每序列可用上下文（非 unified 时被硬性限制为 `n_ctx/N`）；
  - 是否允许跨 stream 拷贝（非 unified 才有 `sc_info` 路径）；
  - `try_clear_idle_slots()`：**只有 unified 才允许真正清空空闲 slot 来复用空间**，非 unified 会提前 return（`tools/server/server-context.cpp:1648-1650`，注释 `[TAG_IDLE_SLOT_CLEAR]` 解释了原因："without a unified KV cache, clearing a slot frees no reusable room"）。

### 8.3 `--cont-batching`

- 见 7.4。关闭后每个 slot 单独成批，KV 的落位更可能连续，但 GPU 利用率下降。

### 8.4 `-c / --ctx-size`、`-ctk/-ctv`

- `-c` 是**所有 slot 的总上下文**（非 unified）或**共享池大小**（unified）。
- `n_ctx` 先被 `GGML_PAD(cparams.n_ctx, 256)` 对齐（`src/llama-context.cpp:284`）。
- `-ctk/-ctv` 决定 `type_k/type_v`，直接线性缩放 KV 字节数（如 `q8_0` 约为 `f16` 的一半）。**量化 V 缓存要求开 FlashAttention**。

---

## 9. 与 vLLM PagedAttention 的对应关系和区别

**先明确：llama.cpp 没有 PagedAttention，也没有任何同名/同义实现。** 下面的"对应"是**功能层面**的类比，不是同名组件的对照。

| 维度 | vLLM PagedAttention | llama.cpp（本仓库） |
|---|---|---|
| 是否叫 PagedAttention | 是，核心机制 | **否**。全仓库无此标识符（仅 `vendor/miniaudio` 有无关的 `paged` 音频缓冲） |
| 物理存储 | 固定大小 **block**（如 16 token）组成的 block pool，物理块可离散 | 每 stream **一整块连续 3D tensor** `[n_embd_k_gqa, kv_size, n_stream]`（`llama-kv-cache.cpp:231-232`） |
| 逻辑→物理映射 | **block table**（每序列一张表，逻辑块 -> 物理块） | **无映射表**。`llama_kv_cells.pos[]` 数组 + `slot_info.idxs[]` 一次性给出本次 token 的 cell 下标 |
| 分配粒度 | 按 block 惰性分配、按需增长 | **一次性按 `kv_size` 全额分配**，运行时只在固定数组内复用槽位 |
| 查找空闲 | 块级 free list | **环形游标 + 线性扫描**（`find_slot`，`llama-kv-cache.cpp:894`，游标 `v_heads`） |
| 内部碎片 | 有（每序列最后一个 block 未用满） | 无块内碎片概念；但**非 unified 的静态等分**会造成"slot 空闲却不能被别人使用"的外部浪费 |
| 前缀共享 | 通过 block 共享 + COW | 通过 cell 的 `seq` **bitset 引用计数**共享（`llama-kv-cells.h:486-489`、`seq_cp` 同 stream 分支 `:459-488`），**无 COW** |
| 注意力读取 | 按 block table gather | **稠密线性扫描**：CUDA FA kernel 在连续 KV 上按 `nbatch_fa` 分块循环，`KV_max` 给每序列上界（`ggml/src/ggml-cuda/fattn-tile.cuh:954`、`:958-977`），非本序列位置靠 **mask 置 -inf** 屏蔽 |
| 显存碎片 | 需要 PagedAttention 解决碎片 | 不需要：**没有动态分配就没有碎片问题**；代价是必须一开始就分配下 `n_ctx` 的全量 KV |

**结论**：vLLM 用"分页 + 块表"换取**高显存利用率和弹性增长**；llama.cpp 用"连续大块 + 槽位复用 + 掩码"换取**实现简单、kernel 友好**。二者解决的不是同一个问题：llama.cpp 的方案在"分配得下"的前提下没有碎片问题，但**无法突破一次性分配的显存上限**——这正是本次 32k 上下文在 8 GiB 显卡上必然 OOM 的原因（见 §11 实测）。

---

## 10. CUDA FlashAttention 路径

- 入口：`ggml_cuda_flash_attn_ext()`，`ggml/src/ggml-cuda/fattn.cu:570`
- 分派（`:576/:579/:582`）：
  - `ggml_cuda_flash_attn_ext_tile()` —— `fattn-tile.cuh`
  - `ggml_cuda_flash_attn_ext_vec()` —— `fattn-vec.cuh`
  - `ggml_cuda_flash_attn_ext_mma_f16()` —— `fattn-mma-f16.cuh`（Ada/Ampere 首选，`ncols1/ncols2` 组合在 `fattn.cu:113-235` 展开）
- 支持的 KV 类型：`fattn.cu:338-351`（F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0）—— 共 8 种，`fattn-vec-instance-*-*.cu` 逐组合实例化。
- K/V 迭代方式（`fattn-tile.cuh:954-977`）：
  ```cpp
  const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;
  while (k_VKQ_0 < k_VKQ_max - nbatch_fa) { flash_attn_tile_iter<...>(...); k_VKQ_0 += nbatch_fa; }
  ```
  即在**连续 KV 上做稠密分块扫描**，`nbatch_fa` 是 compute tiling 参数（`fattn-tile.cuh:12-13` 的 config 宏），**与内存分页无关**。

---

## 11. 本次实测对上述机制的验证（8 GiB RTX 4060 Laptop，Qwen-7B Q4_K_M）

Qwen-7B 是 **fused QKV、无 GQA**（GGUF 元数据只有 `qwen.attention.head_count`，无 `head_count_kv`；tensor 名为 `blk.N.attn_qkv.weight`），
`n_layer = 32`，`n_head = 32`，`head_dim = 128` => `n_embd_k_gqa = 4096`。

- 每 cell（= 每 token）K+V 字节数 = `2 × 32层 × 4096 × 2字节` = **512 KiB**
- `-c 32768 -np 10`（非 unified）：`n_ctx_seq = 32768/10 = 3276 -> pad 3328`，`n_ctx = 33280`
  => `33280 cell × 512 KiB = 16640 MiB` —— 与实测日志**完全一致**：
  > `allocating 16640.00 MiB on device 0: cudaMalloc failed: out of memory`
  > `failed to allocate CUDA0 buffer of size 17448304640`  (17448304640 / 1024² = 16640 MiB)
- `-c 32768 -np 10 --kv-unified`：`kv_size = 32768`, `n_stream = 1`
  => `32768 × 512 KiB = 16384 MiB` —— 同样与实测一致：
  > `allocating 16384.00 MiB ... cudaMalloc failed: out of memory`

日志还验证了 `:296-298` 的对齐逻辑：
> `llama_context: n_ctx is not divisible by n_seq_max - rounding down to 33280`

---

## 12. 关键源码路径与行号索引

### src/llama-kv-cells.h
| 行号 | 内容 |
|---|---|
| `:13-28` | `llama_kv_cell_ext`（M-RoPE 二维位置） |
| `:32` | `class llama_kv_cells` |
| `:222` | `rm(i)` 清空 cell |
| `:238` | `seq_rm(i, seq_id)` 摘除序列位，返回 cell 是否变空 |
| `:261` | `seq_keep(i, seq_id)` |
| `:309` | `seq_add(i, seq_id)` 加序列位（前缀共享） |
| `:334` / `:349` | `seq_pos_min` / `seq_pos_max` |
| `:395` / `:413` / `:442` | `pos_set` / `pos_add` / `pos_div` |
| `:459-499` | 私有字段：`has_shift`,`used`,`pos`,`ext`,`shift`,`seq`,`seq_pos` |
| `:535` | `using llama_kv_cells_vec = std::vector<llama_kv_cells>` |

### src/llama-kv-cache.h
| 行号 | 内容 |
|---|---|
| `:20` | `class llama_kv_cache : public llama_memory_i` |
| `:34-92` | `struct slot_info`（`s0/s1/strm/idxs`，`head()` `:45`，`is_contiguous()` `:77`） |
| `:99-115` | 构造函数签名（含 `unified`, `kv_size`, `n_seq_max`, `n_pad`） |
| `:171` | `get_n_kv()` |
| `:174-179` | `get_k/get_v`、`cpy_k/cpy_v` |
| `:187` | `prepare()` |
| `:189` | `update()` |
| `:194` | `find_slot(ubatch, cont)` |
| `:197` | `apply_ubatch()` |
| `:224-234` | `struct kv_layer` |
| `:238-242` | `n_seq_max` / `n_stream` / `n_pad` |
| `:266` | `ctxs_bufs` |
| `:270` | `v_heads`（环形游标） |
| `:275-277` | `v_cells_impl` / `v_cells` |
| `:280` | `seq_to_stream` |
| `:285-288` | `layers` / `map_layer_ids` |
| `:323` | `class llama_kv_cache_context` |

### src/llama-kv-cache.cpp
| 行号 | 内容 |
|---|---|
| `:64` | 构造函数 |
| `:82` | **`n_stream(unified ? 1 : n_seq_max)`** |
| `:98` | `GGML_ASSERT(kv_size % n_pad == 0)` |
| `:133` | `GGML_ASSERT(n_stream == 1 \|\| n_stream == n_seq_max)` |
| `:140-153` | cell 数组分配 / `seq_to_stream` 建立 |
| `:231-232` | **K/V 3D tensor `[n_embd_*_gqa, kv_size, n_stream]`** |
| `:240-243` | 每 stream 2D view |
| `:274-293` | 按 buft 分配 buffer；`:286` 失败抛异常 |
| `:289` / `:299` | `KV buffer size` / `size = ... (cells, layers, seqs), K/V` 日志 |
| `:379` | `seq_rm` |
| `:447` | `seq_cp`（同 stream `:459-488` / 跨 stream `:490-532`） |
| `:539/566/616` | `seq_keep` / `seq_add` / `seq_div` |
| `:681` | `memory_breakdown()` |
| `:698` | `init_batch()`；`:709` `split_simple` vs `split_equal` |
| `:747` | `prepare()`；`:765` `find_slot(ubatch,false)`；`:793-804` 回滚 |
| `:813` | `update()`；`:823-851` stream 拷贝；`:853-889` K-shift |
| `:894` | **`find_slot()`**；`:1001` 游标启发式；`:1014` `n_test`；`:1038-1057` `can_use`；`:1068-1080` 失败返回 |
| `:1093` | `apply_ubatch()`；`:1127` `pos_set`；`:1137-1139` `seq_add`；`:1146-1161` purge；`:1163-1168` 推进 head |
| `:1171` | `get_can_shift()` |
| `:1227` | `get_n_kv()`（n_kv 启发式 + 256 对齐） |
| `:1243/1263` | `get_k/get_v`（构建视图） |
| `:1295/1330` | `cpy_k/cpy_v` |
| `:1800/1810/1820` | `total_size` / `size_k_bytes` / `size_v_bytes` |

### 其它
| 路径:行号 | 内容 |
|---|---|
| `src/llama-context.cpp:284` | `n_ctx = GGML_PAD(n_ctx, 256)` |
| `src/llama-context.cpp:286-300` | **unified / 非 unified 的 `n_ctx_seq` 计算** |
| `src/llama-context.cpp:302-313` | `n_seq_max` / `n_ctx` / `n_ctx_seq` / `kv_unified` 日志 |
| `src/llama-context.cpp:1406`, `:1718` | `kv_unified ? LLAMA_MAX_SEQ : n_seq_max` |
| `src/llama-model.cpp:2138` | `/* attn_kv_size */ cparams.n_ctx_seq` 传入缓存 |
| `src/llama-cparams.h:8` | `#define LLAMA_MAX_SEQ 256` |
| `src/llama-graph.cpp:33-38` | `n_stream` 与 4D `kq_mask` |
| `common/common.h:574` | `bool kv_unified = false;`（默认值） |
| `common/arg.cpp:1288` | `n_parallel = -1`（auto 默认） |
| `common/arg.cpp:1598-1602` | `-kvu/--kv-unified`、`-no-kvu/--no-kv-unified` |
| `common/arg.cpp:2405-2410` | `-np/--parallel` |
| `tools/server/server.cpp:146-151` | **auto 时 `n_parallel=4` 且 `kv_unified=true`** |
| `tools/server/server-context.cpp:1252`, `:1306` | slot 的 `n_ctx = llama_n_ctx_seq(ctx)` |
| `tools/server/server-context.cpp:1270-1271` | `initializing, n_slots, n_ctx_slot, kv_unified` 日志 |
| `tools/server/server-context.cpp:1648-1650` | 仅 unified 才清理空闲 slot |
| `ggml/src/ggml-cuda/fattn.cu:570` | `ggml_cuda_flash_attn_ext()` 入口；`:576/579/582` 分派 |
| `ggml/src/ggml-cuda/fattn.cu:338-351` | FA 支持的 KV 量化类型 |
| `ggml/src/ggml-cuda/fattn-tile.cuh:954-977` | 稠密 KV 分块扫描（`k_VKQ_max` / `nbatch_fa`） |
