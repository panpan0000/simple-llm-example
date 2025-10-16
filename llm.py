"""
为了从矩阵运算的维度 学习大模型推理原理。
这里用一个最小化的 decoder-only 模型，单头注意力、单层，并演示：
1️⃣ embedding lookup
2️⃣ 计算 Q/K/V
3️⃣ 注意力打分（带因果 mask 的含义说明）
4️⃣ 利用 KV Cache 重用之前的 K/V（增量解码）
5️⃣ 通过 FFN 前馈子层（两层 MLP + ReLU）
6️⃣ 输出下一 token 的 logits

说明：Wq/Wk/Wv/Wo 设为正方形矩阵（embed_dim × embed_dim），这是因为它们在同一嵌入空间内做线性变换，和“序列长度（token 数量）”无关。序列长度可变，由缓存中的 K/V 的条目数决定。
此外，增加 prefill 模式：整段序列一次性前向，使用下三角因果掩码；与逐 token 增量解码对照展示。
"""

import numpy as np
# 提升打印可读性与完整性
np.set_printoptions(precision=3, suppress=True, linewidth=160, threshold=100000)

# 通用 pretty 打印函数
def pmat(name, arr):
    a = np.asarray(arr)
    print(f"--- {name} | shape={a.shape} ---")
    def fmt_float(x):
        xf = float(x)
        # 显示 NaN / 无穷
        if np.isnan(xf):
            return "NaN"
        if not np.isfinite(xf):
            return "INF" if xf > 0 else "-INF"
        # 将巨大（例如掩码用到的 -1e9）显示为 INF/-INF，避免视觉噪音
        if xf <= -1e8:
            return "-INF"
        if xf >= 1e8:
            return "INF"
        return f"{xf:.3f}"
    s = np.array2string(
        a,
        precision=3,
        max_line_width=160,
        separator=' ',
        formatter={'float_kind': fmt_float}
    )
    print(s)

# --- 基础配置 ---
vocab = {"狗":0,"貓":1,"牛":2,"大":3,"中":4,"小":5,"吃":6,"玩":7,"跑":8,"和":9,
         "你":10,"我":11,"他":12,"的":13,"好":14,"在":15}
id2tok = {v:k for k,v in vocab.items()}
embed_dim = 4
ffn_dim = 8  # FFN 中间维度，通常大于 embed_dim，这里取个小值便于演示
np.random.seed(42)

# --- 模型参数 ---
# 注意：这些是同一嵌入空间的线性映射，因此设为正方形矩阵
Wq = np.random.randn(embed_dim, embed_dim)  # 查询的线性映射
Wk = np.random.randn(embed_dim, embed_dim)  # 键的线性映射
Wv = np.random.randn(embed_dim, embed_dim)  # 值的线性映射
Wo = np.random.randn(embed_dim, embed_dim)  # 注意力输出的线性映射（到嵌入空间）

# FFN 前馈子层（两层 MLP）：y -> ReLU(y Wff1) Wff2
Wff1 = np.random.randn(embed_dim, ffn_dim)
Wff2 = np.random.randn(ffn_dim, embed_dim)
Wemb = np.random.randn(len(vocab), embed_dim)  # embedding 矩阵
Wout = np.random.randn(embed_dim, len(vocab))  # 输出层

# --- 输入 ---
tokens = ["小","狗","和","小","貓"]
ids = [vocab[t] for t in tokens]
prefill = True  # True：整段序列一次性计算；False：逐 token 增量解码
max_output_tokens = 2  # 生成的最大 token 数量（默认 2）

# --- KV Cache 初始化 ---
K_cache = []
V_cache = []

print("=== 输入序列 ===")
print("ID=(vocab)", ids)
print("tokens=",tokens)
print()

#print("=== 参数矩阵 ===")
#pmat("Wemb(embedding)", Wemb)
#pmat("Wq", Wq)
#pmat("Wk", Wk)
#pmat("Wv", Wv)
#pmat("Wo", Wo)
#pmat("Wff1", Wff1)
#pmat("Wff2", Wff2)
#pmat("Wout", Wout)

if prefill:
    print("===== Prefill: 整段序列一次性前向 =====")
    T = len(ids)
    # 1️⃣ Embedding：得到整段序列的嵌入矩阵 X ∈ [T, d]
    X = Wemb[ids]
    pmat("X( All input tokens afeter embedding)", X)

    # 2️⃣ 线性映射得 Q/K/V（形状均为 [T, embed_dim]）
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    pmat("Q", Q)
    pmat("K", K)
    pmat("V", V)

    # 3️⃣ 注意力计算：scores ∈ [T, T]
    scores = Q @ K.T / np.sqrt(embed_dim)
    mask = np.tril(np.ones((T, T)))  # 因果掩码：屏蔽未来位置（上三角）
    scores = np.where(mask==1, scores, -1e9)
    pmat("Scores (masked)", scores)
    pmat("Causal mask", mask)
    # softmax（逐行），减去行最大值做数值稳定
    scores_stable = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores_stable)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    pmat("Attention weights", attn)

    # 4️⃣ 聚合得到上下文并线性映射到嵌入空间
    context = attn @ V              # [T, d]
    attn_out = context @ Wo         # [T, d]
    #pmat("Context", context)
    pmat("Attn out", attn_out)

    # 5️⃣ 残差 + FFN（两层 MLP），逐位置处理
    y1 = X + attn_out
    ffn_hidden = np.maximum(0, y1 @ Wff1)
    ffn_out = ffn_hidden @ Wff2
    y2 = y1 + ffn_out
    #pmat("y1 = X + AttnOut", y1)
    pmat("FFN hidden (ReLU)", ffn_hidden)
    #pmat("FFN out", ffn_out)
    pmat("y2 = y1 + FFN", y2)

    # 6️⃣ 输出层得到所有位置的 logits（通常用最后一个位置预测下一个 token）
    logits_all = y2 @ Wout           # [T, |vocab|]
    #pmat("Logits (all positions)", logits_all)        #每一个词汇表每一个词的概率
    pred_id = np.argmax(logits_all[-1])
    print(f"最后位置的下一个 token 预测 → '{id2tok[pred_id]}'")

    # ===== 生成阶段：基于 KV Cache 增量生成接下来的 token =====
    print(f"\n===== Generation: 生成最多 {max_output_tokens} 个 token =====")
    # 初始化 KV cache 为已存在序列的 K/V（逐行切片，保证形状 [1, d]）
    K_cache = [K[i:i+1] for i in range(T)]
    V_cache = [V[i:i+1] for i in range(T)]

    generated_ids = []
    next_id = pred_id  # 第一个要输出的 token（由 prefill 的最后位置产生）
    for i in range(max_output_tokens):
        print("========================生成第",i+1,"个 token========================")
        # 1️⃣ 以上一步预测的 token 作为当前输入，增量计算 Q/K/V
        x = Wemb[next_id].reshape(1, -1)
        Q = x @ Wq
        K_cur = x @ Wk
        V_cur = x @ Wv
        pmat("Gen/x (embedding)", x)
        pmat("Gen/Q", Q)
        pmat("Gen/K_cur", K_cur)
        pmat("Gen/V_cur", V_cur)
        K_cache.append(K_cur)
        V_cache.append(V_cur)
        K_all = np.concatenate(K_cache, axis=0)  # [t, d]
        V_all = np.concatenate(V_cache, axis=0)
        pmat("Gen/K_all (cache)", K_all)
        pmat("Gen/V_all (cache)", V_all)

        # 2️⃣ 注意力（增量步内无未来位置，mask 等效全 1）
        scores = Q @ K_all.T / np.sqrt(embed_dim)  # [1, t]
        pmat("Gen/Scores", scores)
        attn = np.exp(scores - scores.max())
        attn = attn / attn.sum(axis=-1, keepdims=True)
        pmat("Gen/Attention weights", attn)

        # 3️⃣ 聚合、残差、FFN、输出 logits
        context = attn @ V_all
        attn_out = context @ Wo
        y1 = x + attn_out
        ffn_hidden = np.maximum(0, y1 @ Wff1)
        ffn_out = ffn_hidden @ Wff2
        y2 = y1 + ffn_out
        logits = y2 @ Wout
        #pmat("Gen/Context", context)
        #pmat("Gen/Attn out", attn_out)
        #pmat("Gen/y1", y1)
        #pmat("Gen/FFN hidden", ffn_hidden)
        #pmat("Gen/FFN out", ffn_out)
        #pmat("Gen/y2", y2)
        #pmat("Gen/Logits", logits)
        # 下一个 token 的预测（作为后续循环的输入）
        next_id = np.argmax(logits)
        generated_ids.append(next_id)

    print("生成的 tokens:", [id2tok[i] for i in generated_ids])
