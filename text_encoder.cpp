#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

const int SEQ_LEN = 32;         // 句子長度 (Token 數量)
const int HIDDEN_DIM = 128;     // 隱藏層維度 (必須能被 NUM_HEADS 整除)
const int NUM_HEADS = 4;        // 注意力機制的頭數
const int HEAD_DIM = HIDDEN_DIM / NUM_HEADS; 
const int FILTER_DIM = 512;     // FFN 的擴展維度 (通常是 Hidden 的 4 倍)
const int KERNEL_SIZE = 3;      // 1D Conv 的卷積核大小


void init_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
    }
}

// =====================================================================
// [Part 2]: Layer Normalization (Vector Reduction)
// =====================================================================
void layer_norm(const std::vector<float>& input, std::vector<float>& output, int seq_len, int dim) {
    for (int i = 0; i < seq_len; ++i) {
        float sum = 0.0f, sq_sum = 0.0f;
        int offset = i * dim;
        
        // Part 2 Vector Reduction
        for (int j = 0; j < dim; ++j) {
            sum += input[offset + j];
        }
        float mean = sum / dim;
        
        // 第二次 Vector Reduction
        for (int j = 0; j < dim; ++j) {
            float diff = input[offset + j] - mean;
            sq_sum += diff * diff;
        }
        float var = sq_sum / dim;
        
        // 這裡可作為向量化算術運算 (Vector Arithmetic)
        for (int j = 0; j < dim; ++j) {
            output[offset + j] = (input[offset + j] - mean) / std::sqrt(var + 1e-5f);
        }
    }
}

// =====================================================================
// [Part 3 / Part 4]: Multi-Head Attention
// =====================================================================
void multi_head_attention(const std::vector<float>& input, std::vector<float>& output) {

    std::vector<float> Q(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> K(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> V(SEQ_LEN * HIDDEN_DIM);
    init_random(Q); init_random(K); init_random(V);

    float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; ++h) {
        int head_offset = h * HEAD_DIM;
        
        for (int i = 0; i < SEQ_LEN; ++i) {
            std::vector<float> scores(SEQ_LEN, 0.0f);
            float max_score = -1e9f;

            // 1. Q * K^T
            for (int j = 0; j < SEQ_LEN; ++j){
                float dot_product = 0.0f;
                for (int d = 0; d < HEAD_DIM; ++d){
                    dot_product += Q[i * HIDDEN_DIM + head_offset + d] * K[j * HIDDEN_DIM + head_offset + d];
                }
                scores[j] = dot_product * scale;
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

            // 3.  Attention * V
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
// [Part 4 / Part 5]: 1D Convolution
// =====================================================================
void ffn_conv1d(const std::vector<float>& input, std::vector<float>& output, int in_dim, int out_dim) {
    std::vector<float> weights(out_dim * in_dim * KERNEL_SIZE);
    init_random(weights);
    int pad = KERNEL_SIZE / 2;

    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int oc = 0; oc < out_dim; ++oc) {
            float sum = 0.0f;
            
            // Convolution with padding
            for (int k = 0; k < KERNEL_SIZE; ++k) {
                int pos = i + k - pad;
                if (pos >= 0 && pos < SEQ_LEN) { // Padding
                    for (int ic = 0; ic < in_dim; ++ic) {
                        float in_val = input[pos * in_dim + ic];
                        float w_val = weights[oc * (in_dim * KERNEL_SIZE) + ic * KERNEL_SIZE + k];
                        sum += in_val * w_val;
                    }
                }
            }
            // ReLU
            output[i * out_dim + oc] = sum > 0.0f ? sum : 0.0f;
        }
    }
}

// --- 執行一個完整的 VITS Encoder Block ---
void vits_encoder_block(std::vector<float>& x) {
    std::vector<float> norm1_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> attn_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> norm2_out(SEQ_LEN * HIDDEN_DIM);
    std::vector<float> ffn_out(SEQ_LEN * FILTER_DIM);
    std::vector<float> ffn_proj_out(SEQ_LEN * HIDDEN_DIM);

    // 1. 第一層 Layer Normalization
    layer_norm(x, norm1_out, SEQ_LEN, HIDDEN_DIM);
    
    // 2. Multi-Head Attention
    multi_head_attention(norm1_out, attn_out);
    
    // Residual Connection 1
    for (size_t i = 0; i < x.size(); ++i) x[i] += attn_out[i];
    
    // 3. 第二層 Layer Normalization
    layer_norm(x, norm2_out, SEQ_LEN, HIDDEN_DIM);
    
    // 4. FFN: 升維 (1D Conv)
    ffn_conv1d(norm2_out, ffn_out, HIDDEN_DIM, FILTER_DIM);
    
    // 5. FFN: 降維投影 (等效於 kernel_size=1 的 Conv1D 或 Linear)
    // 這裡為保持簡潔，直接用一個簡單的線性轉換模擬降維
    for (int i = 0; i < SEQ_LEN; ++i) {
        for (int j = 0; j < HIDDEN_DIM; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < FILTER_DIM; ++k) {
                sum += ffn_out[i * FILTER_DIM + k] * 0.01f;
            }
            ffn_proj_out[i * HIDDEN_DIM + j] = sum;
        }
    }

    // Residual Connection 2
    for (size_t i = 0; i < x.size(); ++i) x[i] += ffn_proj_out[i];
}

int main() {
    std::vector<float> x(SEQ_LEN * HIDDEN_DIM);
    init_random(x);

    std::cout << "Starting VITS Text Encoder Simulation..." << std::endl;

    vits_encoder_block(x);

    // Checksum OUTPUT
    float checksum = 0.0f;
    for (float val : x) {
        checksum += val;
    }
    
    std::cout << "Encoder Simulation Completed." << std::endl;
    std::cout << "Final Feature Checksum: " << checksum << std::endl;

    return 0;
}
