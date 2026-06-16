#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

const int SEQ_LEN    = 32;
const int HIDDEN_DIM = 128;
const int NUM_HEADS  = 4;
const int HEAD_DIM   = HIDDEN_DIM / NUM_HEADS;
const int FILTER_DIM = 512;
const int KERNEL_SIZE = 3;

#define VLMUL "m1"

static inline size_t linear_proj_strip(const float* in_row,     
                                       const float* w_col_base,  
                                       float* out_ptr,           
                                       size_t remaining_d,       
                                       size_t in_dim) {
    size_t vl;
    size_t stride = in_dim * sizeof(float);   

    asm volatile(
        "vsetvli   %[vl], %[rem], e32, " VLMUL ", ta, ma  \n\t" 
        "vmv.v.i   v8, 0                                    \n\t" 
        "mv        t2, %[indim]                            \n\t" 
        "mv        t3, %[inr]                              \n\t" 
        "mv        t4, %[wbase]                            \n\t" 
        "1:                                                \n\t"
        "flw       ft0, 0(t3)                              \n\t" 
        "vlse32.v  v16, (t4), %[stride]                    \n\t" 
        "vfmacc.vf v8, ft0, v16                            \n\t" 
        "addi      t3, t3, 4                               \n\t" 
        "addi      t4, t4, 4                               \n\t" 
        "addi      t2, t2, -1                              \n\t"
        "bnez      t2, 1b                                  \n\t"
        "vse32.v   v8, (%[out])                            \n\t" 
        : [vl] "=&r"(vl)
        : [rem] "r"(remaining_d), [indim] "r"(in_dim),
          [inr] "r"(in_row), [wbase] "r"(w_col_base),
          [out] "r"(out_ptr), [stride] "r"(stride)
        : "t2", "t3", "t4", "ft0", "memory"
    );
    return vl;
}

void linear_proj_simd(const std::vector<float>& input, const std::vector<float>& W,
                      std::vector<float>& output, int in_dim, int out_dim) {
    for (int i = 0; i < SEQ_LEN; ++i) {
        const float* in_row = &input[i * in_dim];
        int d0 = 0;
        while (d0 < out_dim) {
            const float* w_col_base = &W[(size_t)d0 * in_dim];  
            float* out_ptr = &output[(size_t)i * out_dim + d0];
            size_t vl = linear_proj_strip(in_row, w_col_base, out_ptr,
                                          (size_t)(out_dim - d0), (size_t)in_dim);
            d0 += (int)vl;
        }
    }
}

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
        for (int j = 0; j < dim; ++j) {
            output[offset + j] = (input[offset + j] - mean) * inv_std;
        }
    }
}

void multi_head_attention(const std::vector<float>& input,
                          std::vector<float>& output,
                          const Weights& w) {
    std::vector<float> Q(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> K(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> V(SEQ_LEN * HIDDEN_DIM);

    linear_proj_simd(input, w.Wq, Q, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj_simd(input, w.Wk, K, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj_simd(input, w.Wv, V, HIDDEN_DIM, HIDDEN_DIM);

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
                for (int j = 0; j < SEQ_LEN; ++j) {
                    out_val += scores[j] * V[j * HIDDEN_DIM + head_offset + d];
                }
                output
