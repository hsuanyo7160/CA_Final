#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>

// ---------------------------------------------------------------------
// Hyperparameter
// ---------------------------------------------------------------------
const int SEQ_LEN    = 32;
const int HIDDEN_DIM = 128;
const int NUM_HEADS  = 4;
const int HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;
const int FILTER_DIM = 512;
const int KERNEL_SIZE = 3;


void init_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
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


    init_random(w.Wq);
    init_random(w.Wk);
    init_random(w.Wv);
    init_random(w.conv_w);
    init_random(w.proj_w);
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

    std::vector<float> x(SEQ_LEN * HIDDEN_DIM);
    init_random(x);

    auto st = std::chrono::high_resolution_clock::now();
    vits_encoder_block(x, w);
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = ed - st;

    float sum = 0.0f;
    for (float v : x) sum += v;

    printf("CPU time = %f ms\n", elapsed.count() * 1000.0);
    printf("CPU sum  = %f\n", sum);

    return 0;
}