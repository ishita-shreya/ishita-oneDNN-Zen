/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// onednn_sdpa_alchemist_probe.cpp
// ============================================================
// PURPOSE: Show that the PR #25222 oneDNN Graph SDPA path is numerically
// CORRECT on Battlemage (BMG/Xe2, e.g. Arc Pro B70) but WRONG on
// Alchemist (DG2/XeHPG, e.g. Arc A-series) -- with NO llama.cpp rebuild.
//
// It rebuilds the EXACT SDPA graph that fattn-onednn.cpp::build_sdpa()
// compiles (MatMul->Divide->Add->SoftMax->MatMul, f16 in/out, f32 score),
// runs it on the exact shapes that FAIL test-backend-ops FLASH_ATTN_EXT on
// Alchemist, and compares to a CPU reference using the SAME metric and
// threshold the unit test uses: NMSE = sum((got-ref)^2)/sum(ref^2) < 5e-4
// (ggml test_flash_attn_ext::max_nmse_err).
//
// EXPECTED:
//   B70 (BMG):        all shapes PASS   (systolic sdp_primitive_kernel_t)
//   Arc A / DG2:      shapes FAIL       (different kernel / f16 precision)
//
// To also see WHICH oneDNN kernel is picked (systolic vs larger_partition):
//   ONEDNN_VERBOSE=2 ./alchemist_probe
//
// COMPILE (in the SYCL container, oneDNN 3.11.2):
//   icpx -fsycl -O2 onednn_sdpa_alchemist_probe.cpp
//        -I${DNNLROOT}/include -L${DNNLROOT}/lib -ldnnl
//        -o alchemist_probe && ./alchemist_probe
// ============================================================

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include <sycl/sycl.hpp>

using namespace dnnl;
using namespace dnnl::graph;
using half = sycl::half;

// ---- CPU reference: MHA + GQA scaled-dot-product attention, causal mask ----
// Layout is head-major contiguous, matching the 5D logical tensors:
//   Q [H, q, d]   (head h uses kv-head h/rep),  K,V [Hkv, seq, d],  out [H, q, d]
static std::vector<float> cpu_attention(const std::vector<float> &Q,
        const std::vector<float> &K, const std::vector<float> &V, int H,
        int Hkv, int q, int seq, int d, float scale) {
    const int rep = H / Hkv;
    const int causal_offset = seq - q; // query i sees keys 0..(causal_offset+i)
    std::vector<float> out((size_t)H * q * d, 0.0f);
    std::vector<float> scores(seq);
    for (int h = 0; h < H; ++h) {
        const int hk = h / rep; // kv head for this query head
        const float *Qh = &Q[(size_t)h * q * d];
        const float *Kh = &K[(size_t)hk * seq * d];
        const float *Vh = &V[(size_t)hk * seq * d];
        float *Oh = &out[(size_t)h * q * d];
        for (int i = 0; i < q; ++i) {
            float mx = -1e30f;
            for (int j = 0; j < seq; ++j) {
                float s = 0.0f;
                for (int k = 0; k < d; ++k)
                    s += Qh[i * d + k] * Kh[j * d + k];
                s *= scale;
                if (j > causal_offset + i) s = -INFINITY; // causal mask
                scores[j] = s;
                if (s > mx) mx = s;
            }
            float sum = 0.0f;
            for (int j = 0; j < seq; ++j) {
                scores[j] = (scores[j] == -INFINITY) ? 0.0f
                                                     : std::exp(scores[j] - mx);
                sum += scores[j];
            }
            if (sum == 0.0f) sum = 1.0f;
            for (int k = 0; k < d; ++k) {
                float acc = 0.0f;
                for (int j = 0; j < seq; ++j)
                    acc += (scores[j] / sum) * Vh[j * d + k];
                Oh[i * d + k] = acc;
            }
        }
    }
    return out;
}

struct sdpa_partition {
    compiled_partition cp;
    std::vector<logical_tensor> ins;
    logical_tensor out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0;
    bool ok = false;
};

// Identical graph to fattn-onednn.cpp::build_sdpa().
static sdpa_partition build_sdpa(
        const engine &eng, int mb, int H, int Hkv, int q, int seq, int d) {
    using ltype = logical_tensor::layout_type;
    using dt = logical_tensor::data_type;
    using ldims = logical_tensor::dims;
    const int rep = H / Hkv;
    int64_t id = 0;
    ldims q_sz = {mb, Hkv, rep, q, d}, kv_sz = {mb, Hkv, 1, seq, d},
          s_sz = {mb, Hkv, rep, q, seq};
    sdpa_partition E;

    auto query = logical_tensor(id++, dt::f16, q_sz, ltype::strided);
    auto key = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto score = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    op bmm1(id++, op::kind::MatMul, "bmm1");
    bmm1.set_attr<bool>(op::attr::transpose_b, true);
    bmm1.add_inputs({query, key});
    bmm1.add_outputs({score});

    auto scale = logical_tensor(id++, dt::f16, {1, 1, 1, 1, 1}, ltype::strided);
    auto scaled = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    op sdiv(id++, op::kind::Divide, "scale_div");
    sdiv.add_inputs({score, scale});
    sdiv.add_outputs({scaled});

    auto mask
            = logical_tensor(id++, dt::f16, {mb, 1, 1, q, seq}, ltype::strided);
    auto masked = logical_tensor(id++, dt::f32, s_sz, ltype::strided);
    op madd(id++, op::kind::Add, "mask_add");
    madd.add_inputs({scaled, mask});
    madd.add_outputs({masked});

    auto probs = logical_tensor(id++, dt::f16, s_sz, ltype::strided);
    op smax(id++, op::kind::SoftMax, "softmax");
    smax.set_attr<int64_t>(op::attr::axis, -1);
    smax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
    smax.add_inputs({masked});
    smax.add_outputs({probs});

    auto value = logical_tensor(id++, dt::f16, kv_sz, ltype::strided);
    auto output = logical_tensor(
            id++, dt::f16, q_sz, ltype::strided); // f16 out (systolic path)
    op bmm2(id++, op::kind::MatMul, "bmm2");
    bmm2.add_inputs({probs, value});
    bmm2.add_outputs({output});

    dnnl::graph::graph g(eng.get_kind());
    g.add_op(bmm1);
    g.add_op(sdiv);
    g.add_op(madd);
    g.add_op(smax);
    g.add_op(bmm2);
    g.finalize();
    auto parts = g.get_partitions();
    if (parts.size() != 1 || !parts[0].is_supported()) return E;

    E.ins = parts[0].get_input_ports();
    E.out = parts[0].get_output_ports()[0];
    E.cp = parts[0].compile(E.ins, {E.out}, eng);
    E.out = E.cp.query_logical_tensor(E.out.get_id());
    E.id_q = query.get_id();
    E.id_k = key.get_id();
    E.id_v = value.get_id();
    E.id_scale = scale.get_id();
    E.id_mask = mask.get_id();
    E.ok = true;
    return E;
}

// Run one shape, return true on PASS (NMSE < 5e-4).
static bool run_shape(sycl::queue &sq, const engine &eng, dnnl::stream &strm,
        int H, int Hkv, int q, int seq, int d) {
    const int mb = 1;
    const float scale = 1.0f / std::sqrt((float)d);

    std::mt19937 rng(1234 + d * 131 + H * 17 + q);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hQ((size_t)H * q * d), hK((size_t)Hkv * seq * d),
            hV((size_t)Hkv * seq * d);
    for (auto &x : hQ)
        x = dist(rng);
    for (auto &x : hK)
        x = dist(rng);
    for (auto &x : hV)
        x = dist(rng);

    auto ref = cpu_attention(hQ, hK, hV, H, Hkv, q, seq, d, scale);

    half *dQ = sycl::malloc_device<half>((size_t)H * q * d, sq);
    half *dK = sycl::malloc_device<half>((size_t)Hkv * seq * d, sq);
    half *dV = sycl::malloc_device<half>((size_t)Hkv * seq * d, sq);
    half *dScale = sycl::malloc_device<half>(1, sq);
    half *dMask = sycl::malloc_device<half>((size_t)q * seq, sq);
    half *dOut = sycl::malloc_device<half>((size_t)H * q * d, sq);

    std::vector<half> tQ(hQ.begin(), hQ.end()), tK(hK.begin(), hK.end()),
            tV(hV.begin(), hV.end());
    sq.memcpy(dQ, tQ.data(), tQ.size() * sizeof(half));
    sq.memcpy(dK, tK.data(), tK.size() * sizeof(half));
    sq.memcpy(dV, tV.data(), tV.size() * sizeof(half));
    half hscale = (half)(1.0f / scale);
    sq.memcpy(dScale, &hscale, sizeof(half));

    std::vector<half> hMask((size_t)q * seq);
    const int causal_offset = seq - q;
    for (int i = 0; i < q; ++i)
        for (int j = 0; j < seq; ++j)
            hMask[(size_t)i * seq + j]
                    = (j > causal_offset + i) ? (half)(-65504.0f) : (half)0.0f;
    sq.memcpy(dMask, hMask.data(), hMask.size() * sizeof(half));
    sq.wait();

    auto E = build_sdpa(eng, mb, H, Hkv, q, seq, d);
    if (!E.ok) {
        printf("  d=%-3d rep=%-2d q=%-3d seq=%-4d  |  partition NOT supported "
               "(falls to TILE)\n",
                d, H / Hkv, q, seq);
        sycl::free(dQ, sq);
        sycl::free(dK, sq);
        sycl::free(dV, sq);
        sycl::free(dScale, sq);
        sycl::free(dMask, sq);
        sycl::free(dOut, sq);
        return true; // not the failure mode we're hunting; TILE would handle it
    }

    auto id2ptr = [&](size_t r) -> void * {
        if (r == E.id_q) return dQ;
        if (r == E.id_k) return dK;
        if (r == E.id_v) return dV;
        if (r == E.id_scale) return dScale;
        if (r == E.id_mask) return dMask;
        return nullptr;
    };
    std::vector<tensor> ti;
    ti.reserve(E.ins.size());
    for (auto &lt : E.ins)
        ti.emplace_back(lt, eng, id2ptr(lt.get_id()));
    tensor to(E.out, eng, dOut);
    E.cp.execute(strm, ti, {to});
    strm.wait();

    std::vector<half> hOut((size_t)H * q * d);
    sq.memcpy(hOut.data(), dOut, hOut.size() * sizeof(half)).wait();

    double se = 0.0,
           sref = 0.0; // NMSE = sum(err^2)/sum(ref^2)  (ggml UT metric)
    float max_err = 0.0f, max_ref = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        float got = (float)hOut[i];
        float e = got - ref[i];
        se += (double)e * e;
        sref += (double)ref[i] * ref[i];
        max_err = std::max(max_err, std::fabs(e));
        max_ref = std::max(max_ref, std::fabs(ref[i]));
    }
    double nmse = sref > 0 ? se / sref : se;
    float relmax = max_err / (max_ref > 0 ? max_ref : 1.0f);
    bool pass = nmse < 5e-4;
    printf("  d=%-3d rep=%-2d q=%-3d seq=%-4d  |  NMSE=%.2e  relmax=%.2e  |  "
           "%s\n",
            d, H / Hkv, q, seq, nmse, relmax, pass ? "PASS" : "FAIL");

    sycl::free(dQ, sq);
    sycl::free(dK, sq);
    sycl::free(dV, sq);
    sycl::free(dScale, sq);
    sycl::free(dMask, sq);
    sycl::free(dOut, sq);
    return pass;
}

int main() {
    sycl::queue sq {sycl::gpu_selector_v};
    auto eng = dnnl::sycl_interop::make_engine(
            sq.get_device(), sq.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, sq);

    const dnnl_version_t *v = dnnl_version();
    printf("=== oneDNN SDPA correctness on: %s ===\n",
            sq.get_device().get_info<sycl::info::device::name>().c_str());
    printf("oneDNN version: %d.%d.%d\n", v->major, v->minor, v->patch);
    printf("Threshold: NMSE < 5e-4 (same as test-backend-ops "
           "FLASH_ATTN_EXT)\n");
    printf("Shapes below are the exact ones that FAIL on Alchemist (nh=4, kv "
           "f16, causal).\n\n");

    // Exact failing shapes from PR #25222 UT report (Hkv = nh = 4).
    // {d, rep, q, seq}
    struct Shape {
        int d, rep, q, seq;
    };
    const std::vector<Shape> shapes = {
            {40, 1, 32, 512},
            {40, 4, 32, 512},
            {64, 1, 32, 512},
            {64, 1, 32, 1024},
            {72, 1, 32, 512},
            {80, 1, 32, 512},
            {96, 1, 32, 512},
            {128, 1, 32, 512},
            {128, 4, 32, 512},
            {128, 12, 32, 512},
            {256, 1, 32, 512},
            {512, 1, 32, 512},
    };

    const int Hkv = 4;
    int passed = 0, total = 0;
    for (const auto &s : shapes) {
        ++total;
        passed += run_shape(sq, eng, strm, Hkv * s.rep, Hkv, s.q, s.seq, s.d)
                ? 1
                : 0;
    }
    printf("\n%d/%d shapes PASS.\n", passed, total);
    printf("%s\n",
            passed == total ? ">> SDPA path is correct on this device "
                              "(expected on BMG/Xe2)."
                            : ">> SDPA path is WRONG on this device (expected "
                              "on Alchemist/DG2) -- do NOT enable here.");
    if (passed != total)
        throw std::runtime_error("SDPA path is WRONG on this device.");
    return passed == total ? 0 : 1;
}
