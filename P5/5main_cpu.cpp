// =====================================================================
// VITS Text Encoder - Part 5: CPU 對照版 (main_cpu.cpp)
// Advanced Computer Architecture Final Project
//
// 用途: 與 main.cu (多 pattern GPU 版) 對照
//   - CPU 循序跑 NUM_PATTERNS 次 encoder (與 GPU 平行多 pattern 對照)
//   - 相同確定性測資 -> sum 可直接比對
//   - pattern 0 輸入恆等於 Part 1/4 -> pattern-0 sum 為驗證錨點
//
// 編譯: g++ -O2 main_cpu.cpp -o main_cpu
// =====================================================================

#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

// ---------------------------------------------------------------------
const int SEQ_LEN    = 32;
const int HIDDEN_DIM = 128;
const int NUM_HEADS  = 4;
const int HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;
const int FILTER_DIM = 512;
const int KERNEL_SIZE = 3;

const int NUM_PATTERNS = 8;                       // ★ 與 main.cu 一致 (掃描時兩邊一起改)
const int HID_PER = SEQ_LEN * HIDDEN_DIM;

// ---------------------------------------------------------------------
// 確定性測資生成 (與 main.cu 完全相同)
// ---------------------------------------------------------------------
void init_deterministic(std::vector<float>& vec, int salt) {
    for (size_t i = 0; i < vec.size(); ++i) {
        int t = ((int)i * 13 + 7 + salt * 101) % 271;
        vec[i] = ((float)t / 270.0f) * 0.2f - 0.1f;
    }
}

struct Weights {
    std::vector<float> Wq, Wk, Wv;
    std::vector<float> conv_w;
    std::vector<float> proj_w;
};

void init_weights(Weights& w) {
    w.Wq.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.Wk.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.Wv.resize(HIDDEN_DIM * HIDDEN_DIM);
    w.conv_w.resize(FILTER_DIM * HIDDEN_DIM * KERNEL_SIZE);
    w.proj_w.resize(HIDDEN_DIM * FILTER_DIM);
    init_deterministic(w.Wq,     1);
    init_deterministic(w.Wk,     2);
    init_deterministic(w.Wv,     3);
    init_deterministic(w.conv_w, 4);
    init_deterministic(w.proj_w, 5);
}

// =====================================================================
void layer_norm(const std::vector<float>& input, std::vector<float>& output,
                int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        int offset = i * dim;
        float sum = 0.0f;
        for (int j = 0; j < dim; ++j) sum += input[offset + j];
        float mean = sum / dim;

        float sq_sum = 0.0f;
        for (int j = 0; j < dim; ++j) {
            float diff = input[offset + j] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / dim;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int j = 0; j < dim; ++j)
            output[offset + j] = (input[offset + j] - mean) * inv_std;
    }
}

void linear_proj(const std::vector<float>& input, const std::vector<float>& W,
                 std::vector<float>& output, int in_dim, int out_dim) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int d = 0; d < out_dim; ++d) {
            float s = 0.0f;
            for (int k = 0; k < in_dim; ++k)
                s += input[i * in_dim + k] * W[d * in_dim + k];
            output[i * out_dim + d] = s;
        }
    }
}

void multi_head_attention(const std::vector<float>& input,
                          std::vector<float>& output, const Weights& w) {
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

            for (int j = 0; j < SEQ_LEN; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += Q[i * HIDDEN_DIM + head_offset + d]
                         * K[j * HIDDEN_DIM + head_offset + d];
                }
                scores[j] = dot * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < SEQ_LEN; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (int j = 0; j < SEQ_LEN; ++j) scores[j] /= sum_exp;

            for (int d = 0; d < HEAD_DIM; ++d) {
                float out_val = 0.0f;
                for (int j = 0; j < SEQ_LEN; ++j)
                    out_val += scores[j] * V[j * HIDDEN_DIM + head_offset + d];
                output[i * HIDDEN_DIM + head_offset + d] = out_val;
            }
        }
    }
}

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

    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < FILTER_DIM; ++k)
                sum += ffn_out[i * FILTER_DIM + k] * w.proj_w[j * FILTER_DIM + k];
            ffn_proj_out[i * HIDDEN_DIM + j] = sum;
        }
    }

    for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_proj_out[i];
}

// =====================================================================
int main() {
    Weights w;
    init_weights(w);

    const int TOT_HID = NUM_PATTERNS * HID_PER;
    std::vector<float> h_x(TOT_HID);
    init_deterministic(h_x, 0);   // pattern 0 (前 HID_PER 個) 等於 Part 1/4 的 x

    // 計時: CPU 循序跑 NUM_PATTERNS 次 encoder
    auto st = std::chrono::high_resolution_clock::now();
    for (int p = 0; p < NUM_PATTERNS; ++p) {
        std::vector<float> xp(h_x.begin() + (size_t)p * HID_PER,
                              h_x.begin() + (size_t)(p + 1) * HID_PER);
        vits_encoder_block(xp, w);
        std::copy(xp.begin(), xp.end(), h_x.begin() + (size_t)p * HID_PER);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = ed - st;
    double ms = elapsed.count() * 1000.0;

    float sum = 0.0f;
    for (float v : h_x) sum += v;

    float sum_p0 = 0.0f;
    for (int i = 0; i < HID_PER; ++i) sum_p0 += h_x[i];

    printf("Patterns = %d\n", NUM_PATTERNS);
    printf("CPU time      = %f ms (%f ms / pattern)\n", ms, ms / NUM_PATTERNS);
    printf("CPU sum       = %f\n", sum);
    printf("Pattern-0 sum = %f  (應接近 Part 1)\n", sum_p0);

    return 0;
}
