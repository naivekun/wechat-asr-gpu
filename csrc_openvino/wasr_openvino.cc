// OpenVINO 声学模型。输入 [1,1,T,40]，输出 floor(T/5) 个 CTC token。
#include "../csrc_common/weight_plan.h"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
namespace op = ov::opset13;
using Value = ov::Output<ov::Node>;

Value shape(std::initializer_list<int64_t> dims) {
    return op::Constant::create(ov::element::i64, {dims.size()}, dims);
}

Value weights(const uint8_t* blob, size_t offset, const ov::Shape& dims) {
    return std::make_shared<op::Constant>(ov::element::f32, dims, blob + offset);
}

Value matrix(const uint8_t* blob, size_t offset, float scale, size_t n, size_t k) {
    auto quantized = std::make_shared<op::Constant>(ov::element::i8, ov::Shape{n, k}, blob + offset);
    auto converted = std::make_shared<op::Convert>(quantized, ov::element::f32);
    auto factor = op::Constant::create(ov::element::f32, {}, {scale});
    return std::make_shared<op::Multiply>(converted, factor);
}

Value bias(const Value& x, const uint8_t* blob, size_t offset, size_t n) {
    return std::make_shared<op::Add>(x, weights(blob, offset, {n}));
}

Value norm(const Value& x, const uint8_t* blob, size_t g, size_t b, size_t d) {
    auto y = std::make_shared<op::MVN>(x, shape({-1}), true, 1e-6f, ov::op::MVNEpsMode::INSIDE_SQRT);
    return bias(std::make_shared<op::Multiply>(y, weights(blob, g, {d})), blob, b, d);
}

Value conv(const Value& x, const Value& w, ov::Strides stride, ov::CoordinateDiff pad) {
    auto y = std::make_shared<op::Convolution>(x, w, stride, pad, pad, ov::Strides{1, 1});
    return std::make_shared<op::Relu>(y);
}

std::shared_ptr<ov::Model> graph(const uint8_t* blob, const wl_plan& p, bool masked) {
    auto linear = [&](const Value& x, size_t offset, float scale, size_t n, size_t k) {
        return std::make_shared<op::MatMul>(x, matrix(blob, offset, scale, n, k), false, true);
    };
    const size_t d = p.d, ff = p.ff, flat = p.flat, vocab = p.vocab;
    auto feats = std::make_shared<op::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 40});
    ov::ParameterVector inputs{feats};
    Value frame_mask, token_mask, key_mask;
    if (masked) {
        auto frames = std::make_shared<op::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 1});
        auto tokens = std::make_shared<op::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 1});
        inputs.insert(inputs.end(), {frames, tokens});
        frame_mask = frames;
        token_mask = tokens;
        auto keys = std::make_shared<op::Transpose>(tokens, shape({0, 1, 3, 2}));
        key_mask = std::make_shared<op::Convert>(keys, ov::element::boolean);
    }
    auto c1 = conv(feats, weights(blob, p.c1, {32, 1, 5, 5}), {1, 1}, {2, 2});
    // 静态长度桶的尾部在每个时域卷积后清零，保持与原长度的零边界一致。
    if (masked) {
        c1 = std::make_shared<op::Multiply>(c1, frame_mask);
    }
    auto c2 = conv(c1, weights(blob, p.c2, {64, 32, 7, 7}), {5, 1}, {1, 3});
    if (masked) {
        c2 = std::make_shared<op::Multiply>(c2, token_mask);
    }
    // Rust 的 conv2 只计算前 40 列，第 41 列恒为零，供 conv3 读取。
    auto padded = std::make_shared<op::Pad>(c2, shape({0, 0, 0, 0}), shape({0, 0, 0, 1}),
                                          ov::op::PadMode::CONSTANT);
    auto c3 = conv(padded, weights(blob, p.c3, {192, 64, 3, 3}), {1, 2}, {1, 0});
    auto time_first = std::make_shared<op::Transpose>(c3, shape({0, 2, 1, 3}));
    auto front = std::make_shared<op::Reshape>(time_first, shape({-1, p.flat}), false);
    auto h = bias(linear(front, p.lin, p.lin_s, d, flat), blob, p.lin_b, d);

    auto heads = shape({1, -1, p.heads, p.d / p.heads});
    auto head_first = shape({0, 2, 1, 3});
    auto hidden = shape({-1, p.d});
    for (int i = 0; i < p.n_layers; ++i) {
        const auto& l = p.layers[i];
        auto y = norm(h, blob, l.pn_g, l.pn_b, d);
        auto project = [&](size_t offset, float scale) {
            auto split = std::make_shared<op::Reshape>(linear(y, offset, scale, d, d), heads, false);
            return std::make_shared<op::Transpose>(split, head_first);
        };
        ov::OutputVector qkv{project(l.q, l.qs), project(l.k, l.ks), project(l.v, l.vs)};
        if (masked) {
            qkv.push_back(key_mask);  // padding 位置不参与 softmax
        }
        auto attn = std::make_shared<op::ScaledDotProductAttention>(qkv, false);
        auto merged = std::make_shared<op::Reshape>(
            std::make_shared<op::Transpose>(attn, head_first), hidden, false);
        h = std::make_shared<op::Add>(h, linear(merged, l.o, l.os, d, d));

        y = norm(h, blob, l.mn_g, l.mn_b, d);
        auto f = bias(linear(y, l.f1, l.f1s, ff, d), blob, l.f1b, ff);
        auto gelu = std::make_shared<op::Gelu>(f, ov::op::GeluApproximationMode::TANH);
        auto residual = bias(linear(gelu, l.f2, l.f2s, d, ff), blob, l.f2b, d);
        h = std::make_shared<op::Add>(h, residual);
    }
    auto logits = bias(linear(h, p.out, p.out_s, vocab, d), blob, p.out_b, vocab);
    auto one = op::Constant::create(ov::element::i64, {}, {1});
    auto top = std::make_shared<op::TopK>(logits, one, -1, "max", "value", ov::element::i32, true);
    return std::make_shared<ov::Model>(ov::OutputVector{top->output(1)}, inputs);
}

bool experiment_enabled(const char* name) {
    const auto* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

const std::vector<size_t>& configured_buckets() {
    static const std::vector<size_t> buckets = [] {
        std::vector<size_t> result;
        const char* raw = std::getenv("WTL_NPU_BUCKETS");
        if (!raw || !*raw) {
            return result;
        }
        const char* cursor = raw;
        while (*cursor) {
            char* end = nullptr;
            const auto value = std::strtoull(cursor, &end, 10);
            if (end == cursor || value < 5 || value > 100000) {
                throw std::invalid_argument("WTL_NPU_BUCKETS must contain comma-separated feature lengths");
            }
            result.push_back(static_cast<size_t>(value));
            cursor = end;
            if (*cursor == ',') {
                ++cursor;
            } else if (*cursor) {
                throw std::invalid_argument("WTL_NPU_BUCKETS must contain comma-separated feature lengths");
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }();
    return buckets;
}

size_t static_capacity(size_t frames) {
    const auto& buckets = configured_buckets();
    if (!buckets.empty()) {
        auto it = std::lower_bound(buckets.begin(), buckets.end(), frames);
        if (it != buckets.end()) {
            return *it;
        }
        const auto bucket = buckets.back();
        return ((frames + bucket - 1) / bucket) * bucket;
    }
    return frames <= 500 ? 500 : ((frames + 999) / 1000) * 1000;
}

struct Context {
    ov::Core core;
    const std::string device;
    const size_t vocab;
    const bool static_shapes;
    const bool profile;
    ov::AnyMap properties{
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
    };
    std::shared_ptr<ov::Model> static_graph;
    ov::CompiledModel cpu_model;
    ov::InferRequest cpu_request;
    struct Bucket {
        size_t frames;
        ov::CompiledModel model;
        ov::InferRequest request;
        bool reported;
    };
    std::list<std::shared_ptr<Bucket>> buckets;
    std::mutex infer_mutex;
    std::mutex compile_mutex;
    std::mutex buckets_mutex;

    Context(const uint8_t* blob, const wl_plan& plan, const std::string& device, int threads)
        : device(device), vocab(plan.vocab), static_shapes(device != "CPU"),
          profile(device == "NPU" && experiment_enabled("WTL_NPU_PROFILE")) {
        const bool npu = device == "NPU";
        if (static_shapes && !npu && device != "GPU" && device.rfind("GPU.", 0) != 0) {
            throw std::invalid_argument("supported devices: CPU, GPU, GPU.<index>, NPU");
        }
        properties.insert(ov::hint::inference_precision(npu ? ov::element::f16 : ov::element::f32));
        if (profile) {
            properties.insert(ov::enable_profiling(true));
            std::fprintf(stderr,
                         "NPU_CONFIG\tfull_name=%s\trequested_precision=%s\tplatform=%s\tdriver=%u\tcompiler=%u\ttiles=%lld\n",
                         core.get_property(device, ov::device::full_name).c_str(),
                         properties.at(ov::hint::inference_precision.name()).as<ov::element::Type>().to_string().c_str(),
                         core.get_property(device, ov::intel_npu::platform).c_str(),
                         core.get_property(device, ov::intel_npu::driver_version),
                         core.get_property(device, ov::intel_npu::compiler_version),
                         static_cast<long long>(core.get_property(device, ov::intel_npu::max_tiles)));
        }
        if (const char* cache = std::getenv("WTL_OV_CACHE_DIR"); cache && *cache) {
            properties.insert(ov::cache_dir(cache));
        }
        if (static_shapes) {
            // NPU 使用静态形状。GPU 同样按固定长度编译，以避开动态卷积和 TopK 的问题。
            core.get_property(device, ov::device::full_name);
            static_graph = graph(blob, plan, true);
        } else {
            if (threads > 0) {
                properties.insert(ov::inference_num_threads(threads));
            }
            cpu_model = core.compile_model(graph(blob, plan, false), device, properties);
            cpu_request = cpu_model.create_infer_request();
        }
    }

    std::shared_ptr<Bucket> cached_request(size_t capacity) {
        std::lock_guard<std::mutex> lock(buckets_mutex);
        auto cached = std::find_if(buckets.begin(), buckets.end(), [capacity](const auto& bucket) {
            return bucket->frames == capacity;
        });
        if (cached == buckets.end()) {
            return nullptr;
        }
        buckets.splice(buckets.begin(), buckets, cached);
        return buckets.front();
    }

    std::shared_ptr<Bucket> static_request(size_t capacity) {
        if (auto bucket = cached_request(capacity)) {
            return bucket;
        }
        std::lock_guard<std::mutex> compile_lock(compile_mutex);
        if (auto bucket = cached_request(capacity)) {
            return bucket;
        }
        auto model = static_graph->clone();
        const auto t = static_cast<int64_t>(capacity);
        model->reshape(std::map<size_t, ov::PartialShape>{
            {0, {1, 1, t, 40}},
            {1, {1, 1, t, 1}},
            {2, {1, 1, t / 5, 1}},
        });
        auto compiled = core.compile_model(model, device, properties);
        auto request = compiled.create_infer_request();
        auto bucket = std::make_shared<Bucket>(
            Bucket{capacity, std::move(compiled), std::move(request), false});
        {
            std::lock_guard<std::mutex> lock(buckets_mutex);
            if (buckets.size() >= 3) {
                buckets.pop_back();
            }
            buckets.push_front(bucket);
        }
        return bucket;
    }

    void warm(const size_t* frames, size_t count) {
        if (!static_shapes) {
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            auto bucket = static_request(static_capacity(frames[i]));
            std::vector<float> zeros(bucket->frames * 40);
            std::lock_guard<std::mutex> infer_lock(infer_mutex);
            auto input = bucket->request.get_input_tensor(0);
            std::copy(zeros.begin(), zeros.end(), input.data<float>());
            auto fill_mask = [&](size_t port, size_t size) {
                std::fill_n(bucket->request.get_input_tensor(port).data<float>(), size, 1.f);
            };
            fill_mask(1, bucket->frames);
            fill_mask(2, bucket->frames / 5);
            bucket->request.infer();
            bucket->request.get_output_tensor(0);
        }
    }

    void infer(const float* feats, size_t frames, uint32_t* ids) {
        size_t capacity = frames;
        if (static_shapes) {
            capacity = static_capacity(frames);
        }
        std::shared_ptr<Bucket> bucket;
        std::unique_lock<std::mutex> infer_lock(infer_mutex);
        if (static_shapes) {
            infer_lock.unlock();
            bucket = static_request(capacity);
            infer_lock.lock();
        }
        auto& request = static_shapes ? bucket->request : cpu_request;
        // 请求跨窗口复用，输入由 OpenVINO 持有，避免留下指向 Rust 切片的悬空引用。
        auto input = request.get_input_tensor(0);
        if (!static_shapes) {
            input.set_shape({1, 1, frames, 40});
        }
        auto* data = input.data<float>();
        std::copy_n(feats, frames * 40, data);
        if (static_shapes) {
            std::fill(data + frames * 40, data + capacity * 40, 0.f);
            auto fill_mask = [&](size_t port, size_t valid, size_t size) {
                auto* mask = request.get_input_tensor(port).data<float>();
                std::fill_n(mask, valid, 1.f);
                std::fill(mask + valid, mask + size, 0.f);
            };
            fill_mask(1, frames, capacity);
            fill_mask(2, frames / 5, capacity / 5);
        }
        request.infer();
        if (profile && bucket && !bucket->reported) {
            bucket->reported = true;
            try {
                for (const auto& op : request.get_profiling_info()) {
                    std::fprintf(stderr, "NPU_PROFILE\t%zu\t%s\t%s\t%s\t%lld\n",
                                 capacity, op.node_name.c_str(), op.node_type.c_str(),
                                 op.exec_type.c_str(), static_cast<long long>(op.real_time.count()));
                }
            } catch (const std::exception& e) {
                std::fprintf(stderr, "NPU profiling unavailable: %s\n", e.what());
            }
        }
        auto output = request.get_output_tensor(0);
        if (output.get_size() != capacity / 5) {
            throw std::runtime_error("unexpected OpenVINO token count");
        }
        const auto* tokens = output.data<const int32_t>();
        for (size_t i = 0; i < frames / 5; ++i) {
            if (tokens[i] < 0 || static_cast<size_t>(tokens[i]) >= vocab) {
                throw std::runtime_error("OpenVINO returned an invalid token id");
            }
            ids[i] = tokens[i];
        }
    }
};
}  // namespace

extern "C" void* wasr_ov_create(const uint8_t* blob, const wl_plan* plan, const char* device,
                               int threads, char* error, size_t capacity) noexcept {
    try {
        return new Context(blob, *plan, device, threads);
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return nullptr;
    }
}

extern "C" void wasr_ov_destroy(void* ctx) noexcept {
    delete static_cast<Context*>(ctx);
}

extern "C" bool wasr_ov_forward(void* ctx, const float* feats, size_t frames, uint32_t* ids,
                                char* error, size_t capacity) noexcept {
    try {
        static_cast<Context*>(ctx)->infer(feats, frames, ids);
        return true;
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return false;
    }
}

extern "C" bool wasr_ov_warm(void* ctx, const size_t* frames, size_t count,
                             char* error, size_t capacity) noexcept {
    try {
        static_cast<Context*>(ctx)->warm(frames, count);
        return true;
    } catch (const std::exception& e) {
        std::snprintf(error, capacity, "%s", e.what());
        return false;
    }
}
