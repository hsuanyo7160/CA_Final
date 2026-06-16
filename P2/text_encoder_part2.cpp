// =====================================================================
// VITS Text Encoder - Part 2: RVV Vector Reduction (inline asm)
// Advanced Computer Architecture Final Project
//
// 不使用 <riscv_vector.h>; 全部以 inline asm 撰寫 RVV reduction
// 硬性規定: 使用 vfredusum.vs 將向量元素折疊為純量
//
// 格式參考教授範例:
//   - 操作數以指標傳入 (input operands), 結果用 fsw 寫回
//   - vsetvli 設定 e32/m1/ta/ma
//   - reduction 三步: vmv.v.i v0,0 -> vfredusum.vs -> vfmv.f.s
//   - clobber 列出純量/浮點暫存器 + "memory"
//   - 額外: 加 strip-mining 迴圈處理 n > VLEN 的大向量 (自動分塊)
//
// 編譯 (仍需 -march 讓 assembler 認得向量指令, 但不需 header):
//   riscv64-linux-gnu-g++ -O2 -march=rv64gcv \
//       text_encoder_part2.cpp -o text_encoder_part2
// =====================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------
// 模型超參數
// ---------------------------------------------------------------------
const int SEQ_LEN    = 32;
const int HIDDEN_DIM = 128;
const int NUM_HEADS  = 4;
const int HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;
const int FILTER_DIM = 512;
const int KERNEL_SIZE = 3;

// =====================================================================
// RVV Reduction 核心工具 (inline asm)
//
// 共同結構:
//   [init]  vsetvli 取 vlmax -> vmv.v.i v8,0 (向量累加器歸零)
//   [loop]  vsetvli t0, 剩餘長度 -> 載入 -> 運算 -> 指標前進 -> 遞減
//   [final] vsetvli 取 vlmax -> vfredusum.vs 折疊 -> vfmv.f.s -> fsw
//
// 註: 迴圈用 t2(剩餘) / t3,t4(指標) 為可變副本, 原始 operand 維持唯讀
// =====================================================================

// 純加總: result = Σ a[0..n-1]
static inline float reduce_sum_rvv(const float* a, size_t n) {
    float result = 0.0f;
    asm volatile(
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // vlmax
        "vmv.v.i      v8, 0                        \n\t"  // 累加器歸零
        "mv           t2, %[n]                     \n\t"  // 剩餘元素
        "mv           t3, %[a]                     \n\t"  // 可變指標
        "1:                                        \n\t"
        "vsetvli      t0, t2, e32, m1, ta, ma     \n\t"  // vl = min(剩餘, vlmax)
        "vle32.v      v1, (t3)                     \n\t"  // 載入一塊
        "vfadd.vv     v8, v8, v1                   \n\t"  // 部分和累加
        "slli         t1, t0, 2                    \n\t"  // vl * 4 bytes
        "add          t3, t3, t1                   \n\t"  // 指標前進
        "sub          t2, t2, t0                   \n\t"  // 剩餘遞減
        "bnez         t2, 1b                       \n\t"
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // 折疊用 vlmax
        "vmv.v.i      v0, 0                        \n\t"  // reduction 初始 scalar
        "vfredusum.vs v4, v8, v0                   \n\t"  // v4[0] = Σ v8
        "vfmv.f.s     ft1, v4                       \n\t"  // 取出 element 0
        "fsw          ft1, (%[res])                 \n\t"  // 寫回
        :
        : [a] "r"(a), [n] "r"(n), [res] "r"(&result)
        : "t0", "t1", "t2", "t3", "ft1", "memory"
    );
    return result;
}

// 點積: result = Σ a[k] * b[k]
static inline float dot_rvv(const float* a, const float* b, size_t n) {
    float result = 0.0f;
    asm volatile(
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // vlmax
        "vmv.v.i      v8, 0                        \n\t"  // 累加器歸零
        "mv           t2, %[n]                     \n\t"  // 剩餘元素
        "mv           t3, %[a]                     \n\t"  // 可變指標 a
        "mv           t4, %[b]                     \n\t"  // 可變指標 b
        "1:                                        \n\t"
        "vsetvli      t0, t2, e32, m1, ta, ma     \n\t"  // vl = min(剩餘, vlmax)
        "vle32.v      v1, (t3)                     \n\t"  // 載入 a 一塊
        "vle32.v      v2, (t4)                     \n\t"  // 載入 b 一塊
        "vfmacc.vv    v8, v1, v2                   \n\t"  // 部分積累加 v8 += a*b
        "slli         t1, t0, 2                    \n\t"  // vl * 4 bytes
        "add          t3, t3, t1                   \n\t"
        "add          t4, t4, t1                   \n\t"
        "sub          t2, t2, t0                   \n\t"
        "bnez         t2, 1b                       \n\t"
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // 折疊用 vlmax
        "vmv.v.i      v0, 0                        \n\t"  // reduction 初始 scalar
        "vfredusum.vs v4, v8, v0                   \n\t"  // v4[0] = Σ v8
        "vfmv.f.s     ft1, v4                       \n\t"
        "fsw          ft1, (%[res])                 \n\t"
        :
        : [a] "r"(a), [b] "r"(b), [n] "r"(n), [res] "r"(&result)
        : "t0", "t1", "t2", "t3", "t4", "ft1", "memory"
    );
    return result;
}

// (x - mean)^2 的加總: result = Σ (a[k] - mean)^2
static inline float reduce_sq_diff_rvv(const float* a, float mean, size_t n) {
    float result = 0.0f;
    asm volatile(
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // vlmax
        "vmv.v.i      v8, 0                        \n\t"  // 累加器歸零
        "mv           t2, %[n]                     \n\t"  // 剩餘元素
        "mv           t3, %[a]                     \n\t"  // 可變指標
        "1:                                        \n\t"
        "vsetvli      t0, t2, e32, m1, ta, ma     \n\t"  // vl = min(剩餘, vlmax)
        "vle32.v      v1, (t3)                     \n\t"  // 載入一塊
        "vfsub.vf     v2, v1, %[mean]              \n\t"  // v2 = a - mean
        "vfmacc.vv    v8, v2, v2                   \n\t"  // 累加 (a-mean)^2
        "slli         t1, t0, 2                    \n\t"
        "add          t3, t3, t1                   \n\t"
        "sub          t2, t2, t0                   \n\t"
        "bnez         t2, 1b                       \n\t"
        "vsetvli      t0, zero, e32, m1, ta, ma   \n\t"  // 折疊用 vlmax
        "vmv.v.i      v0, 0                        \n\t"  // reduction 初始 scalar
        "vfredusum.vs v4, v8, v0                   \n\t"  // v4[0] = Σ v8
        "vfmv.f.s     ft1, v4                       \n\t"
        "fsw          ft1, (%[res])                 \n\t"
        :
        : [a] "r"(a), [mean] "f"(mean), [n] "r"(n), [res] "r"(&result)
        : "t0", "t1", "t2", "t3", "ft1", "memory"
    );
    return result;
}

// ---------------------------------------------------------------------
// 權重容器
// ---------------------------------------------------------------------
struct Weights {
    std::vector<float> Wq, Wk, Wv;
    std::vector<float> conv_w;
    std::vector<float> proj_w;
};

void init_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

void init_weights(Weights& w) {
    w.Wq.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.Wk.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.Wv.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.conv_w.resize(FILTER_DIM * HIDDEN_DIM * KERNEL_SIZE);
    w.proj_w.resize(HIDDEN_DIM * FILTER_DIM);
    init_random(w.Wq); init_random(w.Wk); init_random(w.Wv);
    init_random(w.conv_w); init_random(w.proj_w);
}

// =====================================================================
// Layer Normalization  (RVV reduction)
// =====================================================================
void layer_norm(const std::vector<float>& input, std::vector<float>& output,
                int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        int offset = i * dim;
        const float* row = &input[offset];

        float sum = reduce_sum_rvv(row, dim);            // Reduction 1: mean
        float mean = sum / dim;

        float sq_sum = reduce_sq_diff_rvv(row, mean, dim); // Reduction 2: variance
        float var = sq_sum / dim;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < dim; ++j) {
            output[offset + j] = (input[offset + j] - mean) * inv_std;
        }
    }
}

// =====================================================================
// 線性投影 (RVV reduction): out[i][d] = Σ_k in[i][k] * W[d][k]
// =====================================================================
void linear_proj(const std::vector<float>& input, const std::vector<float>& W,
                 std::vector<float>& output, int in_dim, int out_dim) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        const float* in_row = &input[i * in_dim];
        for (int d = 0; d < out_dim; ++d) {
            const float* w_row = &W[d * in_dim];
            output[i * out_dim + d] = dot_rvv(in_row, w_row, in_dim);
        }
    }
}

// =====================================================================
// Multi-Head Attention
//   Q*K^T 用 RVV reduction; Attn*V 暫保留 scalar (strided -> Part 3)
// =====================================================================
void multi_head_attention(const std::vector<float>& input,
                          std::vector<float>& output,
                          const Weights& w) {
    std::vector<float> Q(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> K(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> V(SEQ_LEN * HIDDEN_DIM);

    linear_proj(input, w.Wq, Q, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj(input, w.Wk, K, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj(input, w.Wv, V, HIDDEN_DIM, HIDDEN_DIM);

    float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; ++h) {
        int head_offset = h * HEAD_DIM;

        for (int i = 0; i < SEQ_LEN; ++i) {
            std::vector<float> scores(SEQ_LEN, 0.0f);
            float max_score = -1e9f;

            // 1. Q * K^T (head 內 HEAD_DIM 連續 -> RVV reduction)
            const float* q_ptr = &Q[i * HIDDEN_DIM + head_offset];
            for (int j = 0; j < SEQ_LEN; ++j) {
                const float* k_ptr = &K[j * HIDDEN_DIM + head_offset];
                float dot = dot_rvv(q_ptr, k_ptr, HEAD_DIM);
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j < SEQ_LEN; ++j) {
                scores[j] /= sum_exp;
            }

            // 3. Attn * V (對 j 步進讀 V -> strided, 留 Part 3, 此處 scalar)
            for (int d = 0; d < HEAD_DIM; ++d) {
                float out_val = 0.0f;
                for (int j = 0; j < SEQ_LEN; ++j) {
                    out_val += scores[j] * V[j * HIDDEN_DIM + head_offset + d];
                }
                output[i * HIDDEN_DIM + head_offset + d] = out_val;
            }
        }
    }
}

// =====================================================================
// 1D Convolution (FFN 升維): 保留 scalar (strided weight -> Part 3)
// =====================================================================
void ffn_conv1d(const std::vector<float>& input, std::vector<float>& output,
                const std::vector<float>& weights, int in_dim, int out_dim) {
    int pad = KERNEL_SIZE / 2;
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int oc = 0; oc < out_dim; ++oc) {
            float sum = 0.0f;
            for (int k = 0; k < KERNEL_SIZE; ++k) {
                int pos = i + k - pad;
                if (pos >= 0 && pos < SEQ_LEN) {
                    for (int ic = 0; ic < in_dim; ++ic) {
                        float in_val = input[pos * in_dim + ic];
                        float w_val  = weights[oc * (in_dim * KERNEL_SIZE)
                                             + ic * KERNEL_SIZE + k];
                        sum += in_val * w_val;
                    }
                }
            }
            output[i * out_dim + oc] = (sum > 0.0f) ? sum : 0.0f;
        }
    }
}

// =====================================================================
// 完整 Encoder Block
// =====================================================================
void vits_encoder_block(std::vector<float>& x, const Weights& w) {
    std::vector<float> norm1_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> attn_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> norm2_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> ffn_out(SEQ_LEN * FILTER_DIM);
    std::vector<float> ffn_proj_out(SEQ_LEN * HIDDEN_DIM);

    layer_norm(x, norm1_out, SEQ_LEN, HIDDEN_DIM);
    multi_head_attention(norm1_out, attn_out, w);
    for (size_t i = 0; i < x.size(); ++i) x[i] += attn_out[i];

    layer_norm(x, norm2_out, SEQ_LEN, HIDDEN_DIM);
    ffn_conv1d(norm2_out, ffn_out, w.conv_w, HIDDEN_DIM, FILTER_DIM);

    // FFN 降維投影 (RVV reduction): proj[i][j] = Σ_k ffn[i][k] * proj_w[j][k]
    for (int i = 0; i < SEQ_LEN; ++i) {
        const float* ffn_row = &ffn_out[i * FILTER_DIM];
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            const float* w_row = &w.proj_w[j * FILTER_DIM];
            ffn_proj_out[i * HIDDEN_DIM + j] = dot_rvv(ffn_row, w_row, FILTER_DIM);
        }
    }

    for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_proj_out[i];
}

// =====================================================================
int main() {
    srand(42);

    Weights w;
    init_weights(w);

    std::vector<float> x(SEQ_LEN * HIDDEN_DIM);
    init_random(x);

    std::cout << "Starting VITS Text Encoder Simulation (Part 2 RVV asm)..." << std::endl;

    vits_encoder_block(x, w);

    float checksum = 0.0f;
    for (float val : x) {
        checksum += val;
    }

    std::cout << "Encoder Simulation Completed." << std::endl;
    std::cout << "Final Feature Checksum: " << checksum << std::endl;

    return 0;
}
