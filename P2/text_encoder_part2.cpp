// =====================================================================
// VITS Text Encoder - Part 2: RVV Vector Reduction
// Advanced Computer Architecture Final Project
//
// 硬性規定: 使用 Vector Reduction Operations (vfredusum.vs)
//           將向量暫存器元素折疊為純量
//
// 向量化策略 (本 Part 聚焦「連續記憶體 + reduction」):
//   - layer_norm   : sum / sq_sum  -> RVV reduction
//   - linear_proj  : Q/K/V 投影的 dot product -> RVV reduction
//   - attention    : Q*K^T 的 dot product      -> RVV reduction
//   - ffn 降維投影  : Σ ffn*proj_w               -> RVV reduction
//
//   (Attn*V 與 conv1d 因含 strided access, 刻意留給 Part 3)
//
// 編譯:
//   riscv64-linux-gnu-g++ -O2 -march=rv64gcv \
//       text_encoder_part2.cpp -o text_encoder_part2
//   (GCC >= 13 才有穩定的 RVV v1.0 intrinsics)
// =====================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <riscv_vector.h>

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
// RVV Reduction 核心工具
// =====================================================================

// 純加總: result = Σ a[0..n-1]
//   多個 block 先用 vfadd 累加到向量, 最後一次 vfredusum 折疊為純量
static inline float reduce_sum_rvv(const float* a, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vacc = __riscv_vfadd_vv_f32m1(vacc, va, vl);   // block 部分和
    }
    // 最終 reduction: 把向量累加器折疊成純量
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m1_f32m1(vacc, vzero, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(vred);
}

// 點積: result = Σ a[k] * b[k]
//   用 vfmacc 累加部分積, 最後 vfredusum 折疊
static inline float dot_rvv(const float* a, const float* b, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b + i, vl);
        vacc = __riscv_vfmacc_vv_f32m1(vacc, va, vb, vl);  // 部分積累加
    }
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m1_f32m1(vacc, vzero, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(vred);
}

// (x - mean)^2 的加總: result = Σ (a[k] - mean)^2
//   block 內先 vfsub 再 vfmacc 自乘累加, 最後 vfredusum
static inline float reduce_sq_diff_rvv(const float* a, float mean, size_t n) {
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vacc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a + i, vl);
        vfloat32m1_t vd = __riscv_vfsub_vf_f32m1(va, mean, vl);  // x - mean
        vacc = __riscv_vfmacc_vv_f32m1(vacc, vd, vd, vl);        // 累加 (x-mean)^2
    }
    vfloat32m1_t vzero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t vred  = __riscv_vfredusum_vs_f32m1_f32m1(vacc, vzero, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(vred);
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

        // Reduction 1: mean
        float sum = reduce_sum_rvv(row, dim);
        float mean = sum / dim;

        // Reduction 2: variance
        float sq_sum = reduce_sq_diff_rvv(row, mean, dim);
        float var = sq_sum / dim;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < dim; ++j) {
            output[offset + j] = (input[offset + j] - mean) * inv_std;
        }
    }
}

// =====================================================================
// 線性投影 (RVV reduction): out[i][d] = Σ_k in[i][k] * W[d][k]
//   in 連續, W 同列連續 -> 兩條 unit-stride load, 完美對應 dot_rvv
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

            // 1. Q * K^T  (head 內 HEAD_DIM 連續 -> RVV reduction)
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

            // 3. Attn * V  (對 j 步進讀 V -> strided, 留 Part 3, 此處 scalar)
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
// 1D Convolution (FFN 升維): 保留 scalar
//   weight 對 ic 步進為 KERNEL_SIZE -> strided, 留 Part 3
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

    std::cout << "Starting VITS Text Encoder Simulation (Part 2 RVV)..." << std::endl;

    vits_encoder_block(x, w);

    float checksum = 0.0f;
    for (float val : x) {
        checksum += val;
    }

    std::cout << "Encoder Simulation Completed." << std::endl;
    std::cout << "Final Feature Checksum: " << checksum << std::endl;

    return 0;
}
