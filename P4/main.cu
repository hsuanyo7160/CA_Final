#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Hyperparameter
// ---------------------------------------------------------------------
#define SEQ_LEN     32
#define HIDDEN_DIM  128
#define NUM_HEADS   4
#define HEAD_DIM    (HIDDEN_DIM / NUM_HEADS)
#define FILTER_DIM  512
#define KERNEL_SIZE 3

// Block size 旋鈕 (Part 4: 掃描 occupancy / 效能)
#define BLOCK_SIZE 1024

// ---------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(_e));                                 \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// =====================================================================
// Kernel: Linear Projection
// =====================================================================
__global__ void linear_proj_kernel(const float* __restrict__ in,
                                   const float* __restrict__ W,
                                   float* __restrict__ out,
                                   int in_dim, int out_dim) {
    __shared__ float s_in[FILTER_DIM];        
    int i   = blockIdx.x;           
    int tid = threadIdx.x;

    for (int k = tid; k < in_dim; k += blockDim.x) {
        s_in[k] = in[i * in_dim + k];
    }
    __syncthreads();                          

    for (int d = tid; d < out_dim; d += blockDim.x) {
        const float* w_row = &W[d * in_dim];
        float sum = 0.0f;
        for (int k = 0; k < in_dim; ++k) {
            sum += s_in[k] * w_row[k];         
        }
        out[i * out_dim + d] = sum;
    }
}

// =====================================================================
// Kernel: Q * K^T
// =====================================================================
__global__ void qkt_kernel(const float* __restrict__ Q,
                           const float* __restrict__ K,
                           float* __restrict__ scores, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NUM_HEADS * SEQ_LEN * SEQ_LEN;
    if (idx >= total) return;

    int j = idx % SEQ_LEN;
    int i = (idx / SEQ_LEN) % SEQ_LEN;
    int h = idx / (SEQ_LEN * SEQ_LEN);
    int head_offset = h * HEAD_DIM;

    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
        dot += Q[i * HIDDEN_DIM + head_offset + d]
             * K[j * HIDDEN_DIM + head_offset + d];
    }
    scores[(h * SEQ_LEN + i) * SEQ_LEN + j] = dot * scale;
}

// =====================================================================
// Kernel: Softmax
// =====================================================================
__global__ void softmax_kernel(float* __restrict__ scores) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NUM_HEADS * SEQ_LEN;
    if (idx >= total) return;

    float* row = &scores[idx * SEQ_LEN];

    float max_score = -1e9f;
    for (int j = 0; j < SEQ_LEN; ++j)
        if (row[j] > max_score) max_score = row[j];

    float sum_exp = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        row[j] = expf(row[j] - max_score);
        sum_exp += row[j];
    }
    for (int j = 0; j < SEQ_LEN; ++j)
        row[j] /= sum_exp;
}

// =====================================================================
// Kernel: Attn * V
// =====================================================================
__global__ void av_kernel(const float* __restrict__ scores,
                         const float* __restrict__ V,
                         float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NUM_HEADS * SEQ_LEN * HEAD_DIM;
    if (idx >= total) return;

    int d = idx % HEAD_DIM;
    int i = (idx / HEAD_DIM) % SEQ_LEN;
    int h = idx / (HEAD_DIM * SEQ_LEN);
    int head_offset = h * HEAD_DIM;

    const float* srow = &scores[(h * SEQ_LEN + i) * SEQ_LEN];
    float out_val = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        out_val += srow[j] * V[j * HIDDEN_DIM + head_offset + d];
    }
    out[i * HIDDEN_DIM + head_offset + d] = out_val;
}

// =====================================================================
// Kernel: Layer Normalization
// =====================================================================
__global__ void layer_norm_kernel(const float* __restrict__ in,
                                 float* __restrict__ out, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SEQ_LEN) return;

    int offset = i * dim;
    float sum = 0.0f;
    for (int j = 0; j < dim; ++j) sum += in[offset + j];
    float mean = sum / dim;

    float sq_sum = 0.0f;
    for (int j = 0; j < dim; ++j) {
        float diff = in[offset + j] - mean;
        sq_sum += diff * diff;
    }
    float var = sq_sum / dim;

    float inv_std = rsqrtf(var + 1e-5f);
    for (int j = 0; j < dim; ++j) {
        out[offset + j] = (in[offset + j] - mean) * inv_std;
    }
}

// =====================================================================
// Kernel: 1D Convolution (FFN)
// =====================================================================
__global__ void conv1d_kernel(const float* __restrict__ in,
                            const float* __restrict__ W,
                            float* __restrict__ out,
                            int in_dim, int out_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = SEQ_LEN * out_dim;
    if (idx >= total) return;

    int oc = idx % out_dim;
    int i  = idx / out_dim;
    int pad = KERNEL_SIZE / 2;

    float sum = 0.0f;
    for (int k = 0; k < KERNEL_SIZE; ++k) {
        int pos = i + k - pad;
        if (pos >= 0 && pos < SEQ_LEN) {
            for (int ic = 0; ic < in_dim; ++ic) {
                float in_val = in[pos * in_dim + ic];
                float w_val  = W[oc * (in_dim * KERNEL_SIZE) + ic * KERNEL_SIZE + k];
                sum += in_val * w_val;
            }
        }
    }
    out[i * out_dim + oc] = (sum > 0.0f) ? sum : 0.0f;
}

// =====================================================================
// Kernel: Residual add
// =====================================================================
__global__ void residual_kernel(float* __restrict__ x,
                               const float* __restrict__ y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += y[idx];
}

// ---------------------------------------------------------------------
static inline int grid_for(int total) { return (total + BLOCK_SIZE - 1) / BLOCK_SIZE; }

void init_random(std::vector<float>& v) {
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

// =====================================================================
int main() {
    srand(42);
    std::vector<float> h_Wq(HIDDEN_DIM * HIDDEN_DIM), h_Wk(HIDDEN_DIM * HIDDEN_DIM),
                       h_Wv(HIDDEN_DIM * HIDDEN_DIM);
    std::vector<float> h_conv_w(FILTER_DIM * HIDDEN_DIM * KERNEL_SIZE);
    std::vector<float> h_proj_w(HIDDEN_DIM * FILTER_DIM);
    std::vector<float> h_x(SEQ_LEN * HIDDEN_DIM);

    init_random(h_Wq); init_random(h_Wk); init_random(h_Wv);
    init_random(h_conv_w); init_random(h_proj_w);
    init_random(h_x);

    // --- Device buffers ---
    const int HID = SEQ_LEN * HIDDEN_DIM;
    const int FIL = SEQ_LEN * FILTER_DIM;
    float *d_x, *d_norm1, *d_Q, *d_K, *d_V, *d_scores, *d_attn,
          *d_norm2, *d_ffn, *d_proj;
    float *d_Wq, *d_Wk, *d_Wv, *d_conv_w, *d_proj_w;

    CUDA_CHECK(cudaMalloc(&d_x,      HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm1,  HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q,      HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K,      HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V,      HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, NUM_HEADS * SEQ_LEN * SEQ_LEN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn,   HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm2,  HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ffn,    FIL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj,   HID * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wq,     HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk,     HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv,     HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_w, FILTER_DIM * HIDDEN_DIM * KERNEL_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_proj_w, HIDDEN_DIM * FILTER_DIM * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x,      h_x.data(),      HID * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wq,     h_Wq.data(),     h_Wq.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk,     h_Wk.data(),     h_Wk.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv,     h_Wv.data(),     h_Wv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_conv_w, h_conv_w.data(), h_conv_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_proj_w, h_proj_w.data(), h_proj_w.size() * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    printf("Starting VITS Text Encoder Simulation (Part 4 CUDA SIMT)...\n");

    // ---- Encoder Block ----
    layer_norm_kernel<<<grid_for(SEQ_LEN), BLOCK_SIZE>>>(d_x, d_norm1, HIDDEN_DIM);

    linear_proj_kernel<<<SEQ_LEN, BLOCK_SIZE>>>(d_norm1, d_Wq, d_Q, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj_kernel<<<SEQ_LEN, BLOCK_SIZE>>>(d_norm1, d_Wk, d_K, HIDDEN_DIM, HIDDEN_DIM);
    linear_proj_kernel<<<SEQ_LEN, BLOCK_SIZE>>>(d_norm1, d_Wv, d_V, HIDDEN_DIM, HIDDEN_DIM);

    qkt_kernel<<<grid_for(NUM_HEADS * SEQ_LEN * SEQ_LEN), BLOCK_SIZE>>>(d_Q, d_K, d_scores, scale);
    softmax_kernel<<<grid_for(NUM_HEADS * SEQ_LEN), BLOCK_SIZE>>>(d_scores);
    av_kernel<<<grid_for(NUM_HEADS * SEQ_LEN * HEAD_DIM), BLOCK_SIZE>>>(d_scores, d_V, d_attn);

    residual_kernel<<<grid_for(HID), BLOCK_SIZE>>>(d_x, d_attn, HID);

    layer_norm_kernel<<<grid_for(SEQ_LEN), BLOCK_SIZE>>>(d_x, d_norm2, HIDDEN_DIM);
    conv1d_kernel<<<grid_for(SEQ_LEN * FILTER_DIM), BLOCK_SIZE>>>(d_norm2, d_conv_w, d_ffn, HIDDEN_DIM, FILTER_DIM);
    linear_proj_kernel<<<SEQ_LEN, BLOCK_SIZE>>>(d_ffn, d_proj_w, d_proj, FILTER_DIM, HIDDEN_DIM);

    residual_kernel<<<grid_for(HID), BLOCK_SIZE>>>(d_x, d_proj, HID);

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    std::vector<float> h_out(HID);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, HID * sizeof(float), cudaMemcpyDeviceToHost));

    float checksum = 0.0f;
    for (float v : h_out) checksum += v;

    printf("Encoder Simulation Completed.\n");
    printf("Block size: %d\n", BLOCK_SIZE);
    printf("Kernel time: %.4f ms\n", ms);
    printf("Final Feature Checksum: %f\n", checksum);

    cudaFree(d_x); cudaFree(d_norm1); cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_scores); cudaFree(d_attn); cudaFree(d_norm2); cudaFree(d_ffn); cudaFree(d_proj);
    cudaFree(d_Wq); cudaFree(d_Wk); cudaFree(d_Wv); cudaFree(d_conv_w); cudaFree(d_proj_w);

    return 0;
}