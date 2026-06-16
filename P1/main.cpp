// =====================================================================
// VITS Text Encoder - Scalar Baseline (Part 1)
// Advanced Computer Architecture Final Project
//
// 核心可平行化運算: f(a,b) = a * b, 累加形式 Σ f(a_i, b_i)
//   - Multi-Head Attention (Q*K^T, Attn*V)
//   - FFN Conv1d (升維) + Linear Projection (降維)
//   - LayerNorm (Vector Reduction 對應點)
//
// 防死碼刪除: 最終以 checksum 輸出強迫編譯器保留所有運算
// =====================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// ---------------------------------------------------------------------
// 模型超參數
// ---------------------------------------------------------------------
const int SEQ_LEN    = 32;                    // 句子長度 (Token 數量)
const int HIDDEN_DIM = 128;                   // 隱藏層維度 (須能被 NUM_HEADS 整除)
const int NUM_HEADS  = 4;                     // 注意力機制的頭數
const int HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;
const int FILTER_DIM = 512;                   // FFN 擴展維度 (Hidden 的 4 倍)
const int KERNEL_SIZE = 3;                    // 1D Conv 卷積核大小

// ---------------------------------------------------------------------
// 權重容器 (一次初始化, 固定不變, 確保可重現)
// ---------------------------------------------------------------------
struct Weights {
    std::vector<float> Wq, Wk, Wv;            // [HIDDEN_DIM * HIDDEN_DIM]
    std::vector<float> conv_w;                // [FILTER_DIM * HIDDEN_DIM * KERNEL_SIZE]
    std::vector<float> proj_w;                // [HIDDEN_DIM * FILTER_DIM]
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

    init_random(w.Wq);
    init_random(w.Wk);
    init_random(w.Wv);
    init_random(w.conv_w);
    init_random(w.proj_w);
}

// =====================================================================
// Layer Normalization
//   核心: 兩次 Σ reduction (sum, sq_sum) -> 對應 Part 2 Vector Reduction
// =====================================================================
void layer_norm(const std::vector<float>& input, std::vector<float>& output,
                int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        int offset = i * dim;

        // Reduction 1: mean
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum += input[offset + j];
        }
        float mean = sum / dim;

        // Reduction 2: variance
        float sq_sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = input[offset + j] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / dim;

        // 逐元素正規化 (Vector Arithmetic)
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < dim; ++j) {
            output[offset + j] = (input[offset + j] - mean) * inv_std;
        }
    }
}

// =====================================================================
// 線性投影: out[i][d] = Σ_k in[i][k] * W[d][k]
//   通用於 Q = X*Wq, K = X*Wk, V = X*Wv
//   核心 nested loop: f(a,b) = a*b, 累加成純量 (Reduction 對應點)
// =====================================================================
void linear_proj(const std::vector<float>& input, const std::vector<float>& W,
                 std::vector<float>& output, int in_dim, int out_dim) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < out_dim; ++d) {
            float s = 0.0f;
            for (int k = 0; k < in_dim; ++k) {
                s += input[i * in_dim + k] * W[d * in_dim + k];
            }
            output[i * out_dim + d] = s;
        }
    }
}

// =====================================================================
// Multi-Head Attention
//   1. Q,K,V 線性投影 (來自 input, 確保 checksum 有意義)
//   2. Q*K^T -> scale -> softmax
//   3. Attn*V
// =====================================================================
void multi_head_attention(const std::vector<float>& input,
                          std::vector<float>& output,
                          const Weights& w) {
    std::vector<float> Q(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> K(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> V(SEQ_LEN * HIDDEN_DIM);

    // Q = input * Wq, K = input * Wk, V = input * Wv
    linear_proj(input, w.Wq, Q, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj(input, w.Wk, K, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj(input, w.Wv, V, HIDDEN_DIM, HIDDEN_DIM);

    float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; ++h) {
        int head_offset = h * HEAD_DIM;

        for (int i = 0; i < SEQ_LEN; ++i) {
            std::vector<float> scores(SEQ_LEN, 0.0f);
            float max_score = -1e9f;

            // 1. Q * K^T (核心 dot product: Σ q*k)
            for (int j = 0; j < SEQ_LEN; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q[i * HIDDEN_DIM + head_offset + d]
                         * K[j * HIDDEN_DIM + head_offset + d];
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            // 2. Softmax (數值穩定: 減 max)
            float sum_exp = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j < SEQ_LEN; ++j) {
                scores[j] /= sum_exp;
            }

            // 3. Attn * V (核心 weighted sum: Σ score*v)
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
// 1D Convolution (FFN 升維: HIDDEN_DIM -> FILTER_DIM)
//   核心 nested loop: Σ in*weight, 跨 kernel + input channel
//   ReLU activation
// =====================================================================
void ffn_conv1d(const std::vector<float>& input, std::vector<float>& output,
                const std::vector<float>& weights, int in_dim, int out_dim) {
    int pad = KERNEL_SIZE / 2;

    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int oc = 0; oc < out_dim; ++oc) {
            float sum = 0.0f;

            for (int k = 0; k < KERNEL_SIZE; ++k) {
                int pos = i + k - pad;
                if (pos >= 0 && pos < SEQ_LEN) {          // zero padding
                    for (int ic = 0; ic < in_dim; ++ic) {
                        float in_val = input[pos * in_dim + ic];
                        float w_val  = weights[oc * (in_dim * KERNEL_SIZE)
                                             + ic * KERNEL_SIZE + k];
                        sum += in_val * w_val;
                    }
                }
            }

            // ReLU
            output[i * out_dim + oc] = (sum > 0.0f) ? sum : 0.0f;
        }
    }
}

// =====================================================================
// 完整 VITS Encoder Block
//   LN -> MHA -> Residual -> LN -> FFN(升維) -> Proj(降維) -> Residual
// =====================================================================
void vits_encoder_block(std::vector<float>& x, const Weights& w) {
    std::vector<float> norm1_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> attn_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> norm2_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> ffn_out(SEQ_LEN * FILTER_DIM);
    std::vector<float> ffn_proj_out(SEQ_LEN * HIDDEN_DIM);

    // 1. LayerNorm 1
    layer_norm(x, norm1_out, SEQ_LEN, HIDDEN_DIM);

    // 2. Multi-Head Attention
    multi_head_attention(norm1_out, attn_out, w);

    // Residual 1
    for (size_t i = 0; i < x.size(); ++i) x[i] += attn_out[i];

    // 3. LayerNorm 2
    layer_norm(x, norm2_out, SEQ_LEN, HIDDEN_DIM);

    // 4. FFN 升維 (Conv1d): HIDDEN_DIM -> FILTER_DIM
    ffn_conv1d(norm2_out, ffn_out, w.conv_w, HIDDEN_DIM, FILTER_DIM);

    // 5. FFN 降維投影 (Linear): FILTER_DIM -> HIDDEN_DIM
    //    proj_out[i][j] = Σ_k ffn_out[i][k] * proj_w[j][k]
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < FILTER_DIM; ++k) {
                sum += ffn_out[i * FILTER_DIM + k] * w.proj_w[j * FILTER_DIM + k];
            }
            ffn_proj_out[i * HIDDEN_DIM + j] = sum;
        }
    }

    // Residual 2
    for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_proj_out[i];
}

// =====================================================================
int main() {
    srand(42);  // 固定 seed, 確保 baseline 可重現

    Weights w;
    init_weights(w);

    std::vector<float> x(SEQ_LEN * HIDDEN_DIM);
    init_random(x);

    std::cout << "Starting VITS Text Encoder Simulation..." << std::endl;

    vits_encoder_block(x, w);

    // Checksum 輸出 (防死碼刪除)
    float checksum = 0.0f;
    for (float val : x) {
        checksum += val;
    }

    std::cout << "Encoder Simulation Completed." << std::endl;
    std::cout << "Final Feature Checksum: " << checksum << std::endl;

    return 0;
}
