import numpy as np
import math

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """

    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # 1. Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # 2. Split into heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # 3. Scaled dot-product attention
    scores = np.matmul(Q_heads, K_heads.transpose(0, 1, 3, 2)) / math.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    head_outputs = np.matmul(weights, V_heads)

    # 4. Concatenate heads
    concat = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    # 5. Final linear projection
    output = concat @ W_o

    return output
