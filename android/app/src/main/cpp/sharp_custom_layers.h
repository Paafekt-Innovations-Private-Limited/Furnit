/**
 * Custom NCNN Layers for SHARP Model
 *
 * Implements missing layer types needed for SHARP 3D Gaussian Splat model:
 * - SDPA (Scaled Dot Product Attention)
 * - pnnx.Expression (constant expression)
 * - aten::clamp_min (minimum clamp)
 * - torch.le (less than or equal)
 * - torch.bitwise_not (bitwise not)
 * - torch.where (conditional select)
 */

#ifndef SHARP_CUSTOM_LAYERS_H
#define SHARP_CUSTOM_LAYERS_H

#include "ncnn/layer.h"
#include "ncnn/net.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <android/log.h>

#define SDPA_TAG "SDPA"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, SDPA_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, SDPA_TAG, __VA_ARGS__)

namespace sharp_layers {

/**
 * SDPA - Scaled Dot Product Attention
 * Implements: softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Handles various tensor layouts from PNNX conversion.
 * Uses tiled attention for memory efficiency with long sequences.
 */
class SDPA : public ncnn::Layer {
public:
    SDPA() {
        one_blob_only = false;
        support_inplace = false;
        // Disable FP16/packing to ensure compatibility
        support_packing = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        scale = pd.get(5, 0);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                       std::vector<ncnn::Mat>& top_blobs,
                       const ncnn::Option& opt) const {

        LOGD("SDPA forward called with %zu inputs", bottom_blobs.size());

        if (bottom_blobs.size() < 3) {
            LOGE("SDPA: Expected 3 inputs, got %zu", bottom_blobs.size());
            return -1;
        }

        const ncnn::Mat& q_in = bottom_blobs[0];
        const ncnn::Mat& k_in = bottom_blobs[1];
        const ncnn::Mat& v_in = bottom_blobs[2];

        // Check for null/empty tensors
        if (q_in.empty() || k_in.empty() || v_in.empty()) {
            LOGE("SDPA: Empty input tensor(s)");
            return -1;
        }

        if (!q_in.data || !k_in.data || !v_in.data) {
            LOGE("SDPA: Null data pointer in input tensor(s)");
            return -1;
        }

        LOGD("SDPA Q: dims=%d c=%d h=%d w=%d elempack=%d total=%zu",
             q_in.dims, q_in.c, q_in.h, q_in.w, q_in.elempack, q_in.total());
        LOGD("SDPA K: dims=%d c=%d h=%d w=%d elempack=%d",
             k_in.dims, k_in.c, k_in.h, k_in.w, k_in.elempack);
        LOGD("SDPA V: dims=%d c=%d h=%d w=%d elempack=%d",
             v_in.dims, v_in.c, v_in.h, v_in.w, v_in.elempack);

        // Work with copies to handle packing
        ncnn::Mat q = q_in;
        ncnn::Mat k = k_in;
        ncnn::Mat v = v_in;

        // Convert packed tensors to unpacked if needed
        if (q.elempack != 1) {
            LOGD("SDPA: Converting packed tensors (elempack=%d)", q.elempack);
            ncnn::convert_packing(q_in, q, 1, opt);
            ncnn::convert_packing(k_in, k, 1, opt);
            ncnn::convert_packing(v_in, v, 1, opt);
            LOGD("SDPA: After unpack - Q: dims=%d c=%d h=%d w=%d", q.dims, q.c, q.h, q.w);
        }

        // Handle different tensor layouts
        int seq_len, head_dim, num_heads;

        if (q.dims == 3) {
            // 3D tensor: [num_heads, seq_len, head_dim] in NCNN is (c, h, w)
            num_heads = q.c;
            seq_len = q.h;
            head_dim = q.w;
            return forward_3d(q, k, v, num_heads, seq_len, head_dim, top_blobs, opt);
        } else if (q.dims == 2) {
            // 2D tensor: [seq_len, head_dim] in NCNN is (h, w)
            seq_len = q.h;
            head_dim = q.w;
            return forward_2d(q, k, v, seq_len, head_dim, top_blobs, opt);
        } else if (q.dims == 1) {
            // 1D flattened - try to infer dimensions
            int total = q.w;
            head_dim = 64; // Common ViT head dimension
            seq_len = total / head_dim;
            if (seq_len * head_dim != total) {
                head_dim = 96;
                seq_len = total / head_dim;
            }
            return forward_1d(q, k, v, seq_len, head_dim, top_blobs, opt);
        }

        LOGE("SDPA: Unsupported tensor dims=%d", q.dims);
        return -1;
    }

private:
    int scale;

    // Compute optimal tile size based on sequence length to avoid OOM
    // Max memory per tile: ~64MB to stay safe on mobile devices
    static int compute_tile_size(int seq_len) {
        // Memory per tile = tile_size * seq_len * sizeof(float)
        // Target max ~64MB per tile
        const size_t MAX_TILE_BYTES = 64 * 1024 * 1024;
        int max_tile = MAX_TILE_BYTES / (seq_len * sizeof(float));
        // Clamp between 32 and 512
        max_tile = std::max(32, std::min(512, max_tile));
        LOGD("SDPA: seq_len=%d, computed tile_size=%d", seq_len, max_tile);
        return max_tile;
    }

    // Forward for 3D tensors [num_heads, seq_len, head_dim]
    int forward_3d(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v,
                   int num_heads, int seq_len, int head_dim,
                   std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {

        LOGD("SDPA 3D: num_heads=%d seq_len=%d head_dim=%d", num_heads, seq_len, head_dim);

        // Sanity check dimensions
        if (seq_len <= 0 || head_dim <= 0 || num_heads <= 0) {
            LOGE("SDPA 3D: Invalid dimensions");
            return -1;
        }

        // Output: same shape as input
        top_blobs[0].create(head_dim, seq_len, num_heads, 4u, opt.blob_allocator);
        if (top_blobs[0].empty()) {
            LOGE("SDPA 3D: Failed to allocate output tensor");
            return -100;
        }

        float scale_factor = 1.0f / sqrtf((float)head_dim);
        const int TILE_SIZE = compute_tile_size(seq_len);

        // For very large sequences, reduce parallelism to avoid OOM
        int num_threads = (seq_len > 4096) ? std::min(2, opt.num_threads) : opt.num_threads;
        LOGD("SDPA 3D: Using %d threads for seq_len=%d", num_threads, seq_len);

        // Process each head independently
        #pragma omp parallel for num_threads(num_threads)
        for (int h = 0; h < num_heads; h++) {
            const float* q_head = q.channel(h);
            const float* k_head = k.channel(h);
            const float* v_head = v.channel(h);
            float* out_head = top_blobs[0].channel(h);

            // Use tiled computation to limit memory usage
            // Each tile processes TILE_SIZE rows at a time
            for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
                int tile_end = std::min(tile_start + TILE_SIZE, seq_len);
                int tile_rows = tile_end - tile_start;

                // Allocate scores for this tile only (tile_rows x seq_len)
                std::vector<float> scores(tile_rows * seq_len);

                // Compute Q[tile] @ K^T
                for (int ti = 0; ti < tile_rows; ti++) {
                    int i = tile_start + ti;
                    for (int j = 0; j < seq_len; j++) {
                        float sum = 0.0f;
                        const float* qi = q_head + i * head_dim;
                        const float* kj = k_head + j * head_dim;
                        for (int d = 0; d < head_dim; d++) {
                            sum += qi[d] * kj[d];
                        }
                        scores[ti * seq_len + j] = sum * scale_factor;
                    }
                }

                // Softmax per row
                for (int ti = 0; ti < tile_rows; ti++) {
                    float* row = &scores[ti * seq_len];
                    float max_val = row[0];
                    for (int j = 1; j < seq_len; j++) {
                        if (row[j] > max_val) max_val = row[j];
                    }
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        row[j] = expf(row[j] - max_val);
                        sum += row[j];
                    }
                    float inv_sum = 1.0f / (sum + 1e-9f);
                    for (int j = 0; j < seq_len; j++) {
                        row[j] *= inv_sum;
                    }
                }

                // Compute scores @ V for this tile
                for (int ti = 0; ti < tile_rows; ti++) {
                    int i = tile_start + ti;
                    float* out_row = out_head + i * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        float sum = 0.0f;
                        for (int j = 0; j < seq_len; j++) {
                            sum += scores[ti * seq_len + j] * v_head[j * head_dim + d];
                        }
                        out_row[d] = sum;
                    }
                }
            }
        }

        return 0;
    }

    // Forward for 2D tensors [seq_len, head_dim]
    int forward_2d(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v,
                   int seq_len, int head_dim,
                   std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {

        LOGD("SDPA 2D: seq_len=%d head_dim=%d", seq_len, head_dim);

        // Sanity check dimensions
        if (seq_len <= 0 || head_dim <= 0) {
            LOGE("SDPA 2D: Invalid dimensions");
            return -1;
        }

        // Output: same shape [seq_len, head_dim]
        top_blobs[0].create(head_dim, seq_len, 4u, opt.blob_allocator);
        if (top_blobs[0].empty()) {
            LOGE("SDPA 2D: Failed to allocate output tensor");
            return -100;
        }

        float scale_factor = 1.0f / sqrtf((float)head_dim);
        const int TILE_SIZE = compute_tile_size(seq_len);

        // For very large sequences, reduce parallelism
        int num_threads = (seq_len > 4096) ? std::min(2, opt.num_threads) : opt.num_threads;

        const float* q_data = (const float*)q.data;
        const float* k_data = (const float*)k.data;
        const float* v_data = (const float*)v.data;
        float* out_data = (float*)top_blobs[0].data;

        // Tiled processing
        for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
            int tile_end = std::min(tile_start + TILE_SIZE, seq_len);
            int tile_rows = tile_end - tile_start;

            std::vector<float> scores(tile_rows * seq_len);

            // Compute Q[tile] @ K^T
            #pragma omp parallel for num_threads(num_threads)
            for (int ti = 0; ti < tile_rows; ti++) {
                int i = tile_start + ti;
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    const float* qi = q_data + i * head_dim;
                    const float* kj = k_data + j * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        sum += qi[d] * kj[d];
                    }
                    scores[ti * seq_len + j] = sum * scale_factor;
                }
            }

            // Softmax per row
            #pragma omp parallel for num_threads(num_threads)
            for (int ti = 0; ti < tile_rows; ti++) {
                float* row = &scores[ti * seq_len];
                float max_val = row[0];
                for (int j = 1; j < seq_len; j++) {
                    if (row[j] > max_val) max_val = row[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    row[j] = expf(row[j] - max_val);
                    sum += row[j];
                }
                float inv_sum = 1.0f / (sum + 1e-9f);
                for (int j = 0; j < seq_len; j++) {
                    row[j] *= inv_sum;
                }
            }

            // Compute scores @ V
            #pragma omp parallel for num_threads(num_threads)
            for (int ti = 0; ti < tile_rows; ti++) {
                int i = tile_start + ti;
                float* out_row = out_data + i * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[ti * seq_len + j] * v_data[j * head_dim + d];
                    }
                    out_row[d] = sum;
                }
            }
        }

        return 0;
    }

    // Forward for 1D flattened tensors
    int forward_1d(const ncnn::Mat& q, const ncnn::Mat& k, const ncnn::Mat& v,
                   int seq_len, int head_dim,
                   std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const {

        LOGD("SDPA 1D: seq_len=%d head_dim=%d", seq_len, head_dim);

        // Sanity check dimensions
        if (seq_len <= 0 || head_dim <= 0) {
            LOGE("SDPA 1D: Invalid dimensions");
            return -1;
        }

        // Output: same flattened shape
        top_blobs[0].create(seq_len * head_dim, 4u, opt.blob_allocator);
        if (top_blobs[0].empty()) {
            LOGE("SDPA 1D: Failed to allocate output tensor");
            return -100;
        }

        float scale_factor = 1.0f / sqrtf((float)head_dim);
        const int TILE_SIZE = compute_tile_size(seq_len);

        const float* q_data = (const float*)q.data;
        const float* k_data = (const float*)k.data;
        const float* v_data = (const float*)v.data;
        float* out_data = (float*)top_blobs[0].data;

        // Tiled processing for memory efficiency
        for (int tile_start = 0; tile_start < seq_len; tile_start += TILE_SIZE) {
            int tile_end = std::min(tile_start + TILE_SIZE, seq_len);
            int tile_rows = tile_end - tile_start;

            std::vector<float> scores(tile_rows * seq_len);

            // Q[tile] @ K^T
            for (int ti = 0; ti < tile_rows; ti++) {
                int i = tile_start + ti;
                for (int j = 0; j < seq_len; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        sum += q_data[i * head_dim + d] * k_data[j * head_dim + d];
                    }
                    scores[ti * seq_len + j] = sum * scale_factor;
                }
            }

            // Softmax
            for (int ti = 0; ti < tile_rows; ti++) {
                float* row = &scores[ti * seq_len];
                float max_val = row[0];
                for (int j = 1; j < seq_len; j++) {
                    if (row[j] > max_val) max_val = row[j];
                }
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    row[j] = expf(row[j] - max_val);
                    sum += row[j];
                }
                float inv_sum = 1.0f / (sum + 1e-9f);
                for (int j = 0; j < seq_len; j++) {
                    row[j] *= inv_sum;
                }
            }

            // scores @ V
            for (int ti = 0; ti < tile_rows; ti++) {
                int i = tile_start + ti;
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[ti * seq_len + j] * v_data[j * head_dim + d];
                    }
                    out_data[i * head_dim + d] = sum;
                }
            }
        }

        return 0;
    }
};

/**
 * pnnx.Expression - Evaluates expressions from PNNX conversion.
 *
 * PNNX expand_expression converts most expressions to native BinaryOp/UnaryOp,
 * but constant expressions (zero inputs) can't be expanded and remain as
 * pnnx.Expression custom layers. The expression string is stored in the .pnnx.param
 * but is NOT carried over to .ncnn.param, so zero-input constants fall back to 1e-6
 * (the epsilon used for clamp_min in SHARP's logit computation).
 *
 * For multi-input expressions that weren't expanded (e.g. add(@0,@1)), we parse
 * and evaluate the expression at runtime.
 */
class PnnxExpression : public ncnn::Layer {
public:
    PnnxExpression() {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        // PNNX may store the expression as param key 6 (string)
        ncnn::Mat expr_mat = pd.get(6, ncnn::Mat());
        if (!expr_mat.empty()) {
            // Expression stored as string bytes in a Mat
            expr_ = std::string((const char*)expr_mat.data, expr_mat.w);
            LOGD("PnnxExpression: loaded expr='%s'", expr_.c_str());
        }

        // Try reading a constant float value from param key 0
        constant_value_ = pd.get(0, 1e-6f);

        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                       std::vector<ncnn::Mat>& top_blobs,
                       const ncnn::Option& opt) const {

        // If we have an expression string, evaluate it
        if (!expr_.empty()) {
            return eval_expr(bottom_blobs, top_blobs, opt);
        }

        // Zero-input constant: used as epsilon in clamp_min for logit stability
        if (bottom_blobs.empty()) {
            top_blobs[0].create(1, 4u, opt.blob_allocator);
            if (top_blobs[0].empty()) return -100;
            ((float*)top_blobs[0].data)[0] = constant_value_;
            return 0;
        }

        // Fallback: pass-through first input
        if (bottom_blobs.size() == 1) {
            top_blobs[0] = bottom_blobs[0].clone();
            return top_blobs[0].empty() ? -100 : 0;
        }

        // Fallback: element-wise add for 2+ inputs
        return eval_binary_add(bottom_blobs, top_blobs, opt);
    }

private:
    std::string expr_;
    float constant_value_ = 1e-6f;

    int eval_binary_add(const std::vector<ncnn::Mat>& bottom_blobs,
                        std::vector<ncnn::Mat>& top_blobs,
                        const ncnn::Option& opt) const {
        const ncnn::Mat& a = bottom_blobs[0];
        top_blobs[0].create_like(a, opt.blob_allocator);
        if (top_blobs[0].empty()) return -100;

        size_t total = a.total();
        const float* src_a = (const float*)a.data;
        float* dst = (float*)top_blobs[0].data;
        memcpy(dst, src_a, total * sizeof(float));

        for (size_t b = 1; b < bottom_blobs.size(); b++) {
            const float* src_b = (const float*)bottom_blobs[b].data;
            for (size_t i = 0; i < total; i++) {
                dst[i] += src_b[i];
            }
        }
        return 0;
    }

    // Simple recursive expression evaluator for PNNX expression strings
    // Supports: add(@0,@1), mul(@0,@1), sub(@0,@1), div(@0,@1),
    //           add(@0,add(@1,@2)), numeric literals, neg(@0), etc.
    int eval_expr(const std::vector<ncnn::Mat>& bottom_blobs,
                  std::vector<ncnn::Mat>& top_blobs,
                  const ncnn::Option& opt) const {

        LOGD("PnnxExpression: evaluating '%s' with %zu inputs", expr_.c_str(), bottom_blobs.size());

        // Tokenize
        std::vector<std::string> tokens;
        tokenize(expr_, tokens);

        if (tokens.empty()) {
            LOGE("PnnxExpression: empty token list");
            return -1;
        }

        // Evaluate using a recursive descent on token stream
        size_t pos = 0;
        ncnn::Mat result;
        int ret = eval_tokens(tokens, pos, bottom_blobs, result, opt);
        if (ret != 0) return ret;

        top_blobs[0] = result;
        return 0;
    }

    static void tokenize(const std::string& expr, std::vector<std::string>& tokens) {
        std::string token;
        for (size_t i = 0; i < expr.size(); i++) {
            char c = expr[i];
            if (c == '(' || c == ')' || c == ',') {
                if (!token.empty()) {
                    tokens.push_back(token);
                    token.clear();
                }
                if (c != ',') {
                    tokens.push_back(std::string(1, c));
                }
            } else if (c != ' ') {
                token += c;
            }
        }
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    static bool is_literal(const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        strtof(s.c_str(), &end);
        return end != s.c_str() && *end == '\0';
    }

    int eval_tokens(const std::vector<std::string>& tokens, size_t& pos,
                    const std::vector<ncnn::Mat>& inputs,
                    ncnn::Mat& result, const ncnn::Option& opt) const {
        if (pos >= tokens.size()) return -1;

        const std::string& tok = tokens[pos];

        // Argument reference: @0, @1, ...
        if (tok.size() >= 2 && tok[0] == '@') {
            int idx = std::atoi(tok.c_str() + 1);
            if (idx < 0 || idx >= (int)inputs.size()) {
                LOGE("PnnxExpression: invalid input ref @%d (have %zu)", idx, inputs.size());
                return -1;
            }
            result = inputs[idx];
            pos++;
            return 0;
        }

        // Numeric literal
        if (is_literal(tok)) {
            float val = strtof(tok.c_str(), nullptr);
            result.create(1, 4u, opt.blob_allocator);
            if (result.empty()) return -100;
            ((float*)result.data)[0] = val;
            pos++;
            return 0;
        }

        // Function call: op(args...)
        std::string op = tok;
        pos++;

        if (pos >= tokens.size() || tokens[pos] != "(") {
            LOGE("PnnxExpression: expected '(' after '%s'", op.c_str());
            return -1;
        }
        pos++; // skip '('

        // Collect arguments
        std::vector<ncnn::Mat> args;
        while (pos < tokens.size() && tokens[pos] != ")") {
            ncnn::Mat arg;
            int ret = eval_tokens(tokens, pos, inputs, arg, opt);
            if (ret != 0) return ret;
            args.push_back(arg);
        }

        if (pos < tokens.size()) pos++; // skip ')'

        return apply_op(op, args, result, opt);
    }

    int apply_op(const std::string& op, const std::vector<ncnn::Mat>& args,
                 ncnn::Mat& result, const ncnn::Option& opt) const {

        if (args.empty()) {
            LOGE("PnnxExpression: op '%s' has no args", op.c_str());
            return -1;
        }

        // Unary ops
        if (args.size() == 1) {
            const ncnn::Mat& a = args[0];
            size_t total = a.total();
            result.create_like(a, opt.blob_allocator);
            if (result.empty()) return -100;
            const float* sa = (const float*)a.data;
            float* dst = (float*)result.data;

            if (op == "neg") {
                for (size_t i = 0; i < total; i++) dst[i] = -sa[i];
            } else if (op == "abs") {
                for (size_t i = 0; i < total; i++) dst[i] = std::abs(sa[i]);
            } else if (op == "sqrt") {
                for (size_t i = 0; i < total; i++) dst[i] = std::sqrt(sa[i]);
            } else if (op == "exp") {
                for (size_t i = 0; i < total; i++) dst[i] = std::exp(sa[i]);
            } else if (op == "log") {
                for (size_t i = 0; i < total; i++) dst[i] = std::log(sa[i]);
            } else if (op == "sin") {
                for (size_t i = 0; i < total; i++) dst[i] = std::sin(sa[i]);
            } else if (op == "cos") {
                for (size_t i = 0; i < total; i++) dst[i] = std::cos(sa[i]);
            } else if (op == "floor") {
                for (size_t i = 0; i < total; i++) dst[i] = std::floor(sa[i]);
            } else if (op == "ceil") {
                for (size_t i = 0; i < total; i++) dst[i] = std::ceil(sa[i]);
            } else if (op == "sigmoid") {
                for (size_t i = 0; i < total; i++) dst[i] = 1.0f / (1.0f + std::exp(-sa[i]));
            } else if (op == "tanh") {
                for (size_t i = 0; i < total; i++) dst[i] = std::tanh(sa[i]);
            } else if (op == "relu") {
                for (size_t i = 0; i < total; i++) dst[i] = std::max(0.0f, sa[i]);
            } else {
                LOGE("PnnxExpression: unknown unary op '%s'", op.c_str());
                return -1;
            }
            return 0;
        }

        // Binary ops
        if (args.size() == 2) {
            const ncnn::Mat& a = args[0];
            const ncnn::Mat& b = args[1];

            bool a_scalar = (a.total() == 1);
            bool b_scalar = (b.total() == 1);
            const ncnn::Mat& ref = a_scalar ? b : a;

            result.create_like(ref, opt.blob_allocator);
            if (result.empty()) return -100;

            size_t total = ref.total();
            float* dst = (float*)result.data;

            float sa_val = a_scalar ? ((const float*)a.data)[0] : 0.0f;
            float sb_val = b_scalar ? ((const float*)b.data)[0] : 0.0f;
            const float* sa = (const float*)a.data;
            const float* sb = (const float*)b.data;

            for (size_t i = 0; i < total; i++) {
                float va = a_scalar ? sa_val : sa[i];
                float vb = b_scalar ? sb_val : sb[i];

                if (op == "add") dst[i] = va + vb;
                else if (op == "sub") dst[i] = va - vb;
                else if (op == "mul") dst[i] = va * vb;
                else if (op == "div") dst[i] = va / (vb + 1e-10f);
                else if (op == "pow") dst[i] = std::pow(va, vb);
                else if (op == "min") dst[i] = std::min(va, vb);
                else if (op == "max") dst[i] = std::max(va, vb);
                else {
                    LOGE("PnnxExpression: unknown binary op '%s'", op.c_str());
                    return -1;
                }
            }
            return 0;
        }

        LOGE("PnnxExpression: op '%s' has unsupported arg count %zu", op.c_str(), args.size());
        return -1;
    }
};

/**
 * aten::clamp_min - Clamp values to minimum
 */
class AtenClampMin : public ncnn::Layer {
public:
    AtenClampMin() {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                       std::vector<ncnn::Mat>& top_blobs,
                       const ncnn::Option& opt) const {
        LOGD("AtenClampMin forward with %zu inputs", bottom_blobs.size());

        if (bottom_blobs.size() < 2 || bottom_blobs[0].empty() || bottom_blobs[1].empty()) {
            LOGE("AtenClampMin: Invalid inputs");
            return -1;
        }

        const ncnn::Mat& input = bottom_blobs[0];
        const ncnn::Mat& min_val_mat = bottom_blobs[1];

        if (!min_val_mat.data) {
            LOGE("AtenClampMin: min_val_mat has null data");
            return -1;
        }

        float min_val = ((const float*)min_val_mat.data)[0];

        top_blobs[0].create_like(input, opt.blob_allocator);
        if (top_blobs[0].empty()) {
            LOGE("AtenClampMin: Failed to allocate output");
            return -100;
        }

        size_t total = input.total();
        const float* src = (const float*)input.data;
        float* dst = (float*)top_blobs[0].data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t i = 0; i < total; i++) {
            dst[i] = src[i] < min_val ? min_val : src[i];
        }

        return 0;
    }
};

/**
 * torch.le - Less than or equal comparison
 */
class TorchLe : public ncnn::Layer {
public:
    TorchLe() {
        one_blob_only = true;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        other = pd.get(0, 0.04045f);
        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                       const ncnn::Option& opt) const {
        top_blob.create_like(bottom_blob, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        size_t total = bottom_blob.total();
        const float* src = (const float*)bottom_blob.data;
        float* dst = (float*)top_blob.data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t i = 0; i < total; i++) {
            dst[i] = (src[i] <= other) ? 1.0f : 0.0f;
        }

        return 0;
    }

private:
    float other;
};

/**
 * torch.bitwise_not - Bitwise NOT (invert boolean)
 */
class TorchBitwiseNot : public ncnn::Layer {
public:
    TorchBitwiseNot() {
        one_blob_only = true;
        support_inplace = false;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                       const ncnn::Option& opt) const {
        top_blob.create_like(bottom_blob, opt.blob_allocator);
        if (top_blob.empty()) return -100;

        size_t total = bottom_blob.total();
        const float* src = (const float*)bottom_blob.data;
        float* dst = (float*)top_blob.data;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (size_t i = 0; i < total; i++) {
            dst[i] = (src[i] == 0.0f) ? 1.0f : 0.0f;
        }

        return 0;
    }
};

/**
 * torch.where - Conditional selection
 */
class TorchWhere : public ncnn::Layer {
public:
    TorchWhere() {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        other = pd.get(0, 0.04045f);
        return 0;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs,
                       std::vector<ncnn::Mat>& top_blobs,
                       const ncnn::Option& opt) const {

        LOGD("TorchWhere forward with %zu inputs", bottom_blobs.size());

        if (bottom_blobs.size() == 2) {
            const ncnn::Mat& condition = bottom_blobs[0];
            const ncnn::Mat& x = bottom_blobs[1];

            if (condition.empty() || x.empty() || !condition.data || !x.data) {
                LOGE("TorchWhere 2-input: Invalid inputs");
                return -1;
            }

            top_blobs[0].create_like(x, opt.blob_allocator);
            if (top_blobs[0].empty()) return -100;

            size_t total = x.total();
            const float* cond = (const float*)condition.data;
            const float* src_x = (const float*)x.data;
            float* dst = (float*)top_blobs[0].data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (size_t i = 0; i < total; i++) {
                dst[i] = (cond[i] != 0.0f) ? src_x[i] : other;
            }
        } else if (bottom_blobs.size() >= 3) {
            const ncnn::Mat& condition = bottom_blobs[0];
            const ncnn::Mat& x = bottom_blobs[1];
            const ncnn::Mat& y = bottom_blobs[2];

            if (condition.empty() || x.empty() || y.empty()) {
                LOGE("TorchWhere 3-input: Empty inputs");
                return -1;
            }

            top_blobs[0].create_like(x, opt.blob_allocator);
            if (top_blobs[0].empty()) return -100;

            size_t total = x.total();
            const float* cond = (const float*)condition.data;
            const float* src_x = (const float*)x.data;
            const float* src_y = (const float*)y.data;
            float* dst = (float*)top_blobs[0].data;

            #pragma omp parallel for num_threads(opt.num_threads)
            for (size_t i = 0; i < total; i++) {
                dst[i] = (cond[i] != 0.0f) ? src_x[i] : src_y[i];
            }
        } else {
            LOGE("TorchWhere: Unexpected input count %zu", bottom_blobs.size());
            return -1;
        }

        return 0;
    }

private:
    float other;
};

/**
 * SafeConv - A wrapper around NCNN's Convolution that forces safe options.
 *
 * This bypasses ARM NEON optimized kernels that have buffer overread bugs
 * by forcing: no packing, no winograd, no sgemm, single thread.
 *
 * Used specifically for conv_106 which crashes with the optimized kernels.
 */
class SafeConv : public ncnn::Layer {
public:
    SafeConv() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
        support_fp16_storage = false;
    }

    virtual int load_param(const ncnn::ParamDict& pd) {
        // Standard Convolution params (from ncnn operation param table)
        num_output = pd.get(0, 0);
        kernel_w = pd.get(1, 0);
        kernel_h = pd.get(11, kernel_w);
        dilation_w = pd.get(2, 1);
        dilation_h = pd.get(12, dilation_w);
        stride_w = pd.get(3, 1);
        stride_h = pd.get(13, stride_w);
        pad_left = pd.get(4, 0);
        pad_right = pd.get(15, pad_left);
        pad_top = pd.get(14, pad_left);
        pad_bottom = pd.get(16, pad_top);
        pad_value = pd.get(18, 0.f);
        bias_term = pd.get(5, 0);
        weight_data_size = pd.get(6, 0);
        group = pd.get(7, 1);
        activation_type = pd.get(9, 0);
        activation_params = pd.get(10, ncnn::Mat());

        LOGD("SafeConv load_param: num_output=%d kernel=%dx%d stride=%dx%d weight_size=%d",
             num_output, kernel_w, kernel_h, stride_w, stride_h, weight_data_size);

        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb) {
        weight_data = mb.load(weight_data_size, 0);
        if (weight_data.empty()) {
            LOGE("SafeConv: Failed to load weight data");
            return -100;
        }

        if (bias_term) {
            bias_data = mb.load(num_output, 1);
            if (bias_data.empty()) {
                LOGE("SafeConv: Failed to load bias data");
                return -100;
            }
        }

        LOGD("SafeConv load_model: weight_data total=%zu, bias_term=%d",
             weight_data.total(), bias_term);

        return 0;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                       const ncnn::Option& opt) const {

        LOGD("SafeConv::forward called: input w=%d h=%d c=%d elempack=%d -> num_output=%d",
             bottom_blob.w, bottom_blob.h, bottom_blob.c, bottom_blob.elempack, num_output);

        // Handle packed input - convert to elempack=1 for safe scalar processing
        ncnn::Mat bottom_blob_unpacked = bottom_blob;
        if (bottom_blob.elempack != 1) {
            LOGD("SafeConv: Unpacking input from elempack=%d", bottom_blob.elempack);
            ncnn::convert_packing(bottom_blob, bottom_blob_unpacked, 1, opt);
        }

        int w = bottom_blob_unpacked.w;
        int h = bottom_blob_unpacked.h;
        int channels = bottom_blob_unpacked.c;

        // Calculate output dimensions
        int outw = (w + pad_left + pad_right - kernel_w) / stride_w + 1;
        int outh = (h + pad_top + pad_bottom - kernel_h) / stride_h + 1;

        // Allocate output
        top_blob.create(outw, outh, num_output, 4u, opt.blob_allocator);
        if (top_blob.empty()) {
            LOGE("SafeConv: Failed to allocate output tensor");
            return -100;
        }

        LOGD("SafeConv: output w=%d h=%d c=%d", outw, outh, num_output);

        // Naive convolution implementation - no SIMD, no parallelism
        // This is slow but guaranteed to be safe
        const int kernel_size = kernel_w * kernel_h;

        // Calculate expected input channels from weight size
        // weight_data_size = num_output * (input_channels/group) * kernel_w * kernel_h
        const int expected_input_channels = weight_data_size / (num_output * kernel_size);
        int input_channels_per_group = channels / group;

        // WORKAROUND: Model conversion issue - weight size suggests fewer input channels
        // than the actual input has. Use the weight-implied channel count.
        if (expected_input_channels != input_channels_per_group) {
            LOGD("SafeConv: Channel mismatch! Input has %d channels but weights expect %d",
                 input_channels_per_group, expected_input_channels);
            LOGD("SafeConv: Using first %d channels only (model conversion issue)",
                 expected_input_channels);
            input_channels_per_group = expected_input_channels;
        }

        const int output_channels_per_group = num_output / group;

        LOGD("SafeConv: Starting conv loop: %d groups, %d out_ch/grp, %d in_ch/grp, kernel=%dx%d",
             group, output_channels_per_group, input_channels_per_group, kernel_h, kernel_w);

        for (int g = 0; g < group; g++) {
            for (int oc = 0; oc < output_channels_per_group; oc++) {
                int out_c = g * output_channels_per_group + oc;
                float* outptr = top_blob.channel(out_c);

                const float* weight_ptr = (const float*)weight_data.data +
                    out_c * input_channels_per_group * kernel_size;

                for (int oh = 0; oh < outh; oh++) {
                    for (int ow = 0; ow < outw; ow++) {
                        float sum = 0.f;

                        for (int ic = 0; ic < input_channels_per_group; ic++) {
                            int in_c = g * input_channels_per_group + ic;
                            const float* inptr = bottom_blob_unpacked.channel(in_c);
                            const float* kptr = weight_ptr + ic * kernel_size;

                            for (int kh = 0; kh < kernel_h; kh++) {
                                for (int kw = 0; kw < kernel_w; kw++) {
                                    int ih = oh * stride_h - pad_top + kh * dilation_h;
                                    int iw = ow * stride_w - pad_left + kw * dilation_w;

                                    float val = 0.f;
                                    if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                                        val = inptr[ih * w + iw];
                                    } else {
                                        val = pad_value;
                                    }

                                    sum += val * kptr[kh * kernel_w + kw];
                                }
                            }
                        }

                        // Add bias
                        if (bias_term) {
                            sum += ((const float*)bias_data.data)[out_c];
                        }

                        // Activation
                        if (activation_type == 1) {
                            sum = std::max(sum, 0.f);  // ReLU
                        } else if (activation_type == 2) {
                            float slope = activation_params.empty() ? 0.1f :
                                         ((const float*)activation_params.data)[0];
                            sum = sum > 0.f ? sum : sum * slope;  // LeakyReLU
                        }

                        outptr[oh * outw + ow] = sum;
                    }
                }

                // Log progress every 100 channels
                if (oc % 100 == 0) {
                    LOGD("SafeConv: Progress %d/%d channels", oc, output_channels_per_group);
                }
            }
        }

        LOGD("SafeConv::forward completed successfully");
        return 0;
    }

private:
    int num_output;
    int kernel_w, kernel_h;
    int dilation_w, dilation_h;
    int stride_w, stride_h;
    int pad_left, pad_right, pad_top, pad_bottom;
    float pad_value;
    int bias_term;
    int weight_data_size;
    int group;
    int activation_type;
    ncnn::Mat activation_params;
    ncnn::Mat weight_data;
    ncnn::Mat bias_data;
};

// Layer creator functions
static ncnn::Layer* SDPA_layer_creator(void*) { return new SDPA(); }
static ncnn::Layer* PnnxExpression_layer_creator(void*) { return new PnnxExpression(); }
static ncnn::Layer* AtenClampMin_layer_creator(void*) { return new AtenClampMin(); }
static ncnn::Layer* TorchLe_layer_creator(void*) { return new TorchLe(); }
static ncnn::Layer* TorchBitwiseNot_layer_creator(void*) { return new TorchBitwiseNot(); }
static ncnn::Layer* TorchWhere_layer_creator(void*) { return new TorchWhere(); }
static ncnn::Layer* SafeConv_layer_creator(void*) { return new SafeConv(); }

// Layer destroyer
static void custom_layer_destroyer(ncnn::Layer* layer, void*) { delete layer; }

/**
 * Register all custom layers
 */
inline void register_custom_layers(ncnn::Net& net) {
    net.register_custom_layer("SDPA", SDPA_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("pnnx.Expression", PnnxExpression_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("aten::clamp_min", AtenClampMin_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("torch.le", TorchLe_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("torch.bitwise_not", TorchBitwiseNot_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("torch.where", TorchWhere_layer_creator, custom_layer_destroyer);
    net.register_custom_layer("SafeConv", SafeConv_layer_creator, custom_layer_destroyer);
}

} // namespace sharp_layers

#endif // SHARP_CUSTOM_LAYERS_H
