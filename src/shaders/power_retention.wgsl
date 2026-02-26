struct View {
    shape: vec4<u32>,
    stride: vec4<u32>,
    offset: vec4<u32>,
};

struct Cursor {
    batch: u32,
    token: u32,
    len: u32,
};

struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
};

// shape from Q tensor meta: [NUM_HEADS * head_dim, num_token, 1, 1]
@group(0) @binding(0) var<uniform> shape: vec4<u32>;
// view for state tensor: [kv_dim, D, batch, 1]
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var<storage, read> cursors: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>;

#ifdef FP16
@group(0) @binding(4) var<storage, read> gate: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> q: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read> k: array<vec2<u32>>;
@group(0) @binding(7) var<storage, read> v: array<vec2<u32>>;
@group(0) @binding(8) var<storage, read_write> output: array<vec2<u32>>;
#else
@group(0) @binding(4) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> output: array<vec4<f32>>;
#endif

@group(0) @binding(9) var<storage, read_write> sum_of_keys: array<f32>;

// Shared memory for K values (all head_dim scalars), Q values per Q-head group, and normalizers.
var<workgroup> shared_k: array<f32, HEAD_DIM>;
var<workgroup> shared_q: array<f32, GROUP_RATIO * HEAD_DIM>;
var<workgroup> shared_l: array<f32, GROUP_RATIO>;

fn compute_index(batch: u32, token: u32, index: u32) -> u32 {
    let stride = view.stride.x >> 2u;
    let offset = vec3<u32>(view.offset.zy, view.offset.x >> 2u);
    return dot(vec3<u32>(batch, token, index) + offset, vec3<u32>(view.stride.y * stride, stride, 1u));
}

fn compute_cursor(c: u32) -> Cursor {
    var cursor: Cursor;
    cursor.batch = c & 0xffu;
    cursor.token = (c >> 8u) & 0xffffu;
    cursor.len = (c >> 24u) & 0xffu;
    return cursor;
}

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn load_gate(token: u32, head: u32) -> f32 {
    // Map KV head to gate head.
    let gate_head = head * NUM_GATE_HEADS / NUM_KV_HEADS;
    let flat = token * GATE_STRIDE + gate_head;
    let vec_idx = flat / 4u;
    let component = flat % 4u;
#ifdef FP16
    let values = unpack4x16float(gate[vec_idx]);
#else
    let values = gate[vec_idx];
#endif
    return values[component];
}

fn load_q_vec4(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(q[index]);
#else
    return q[index];
#endif
}

fn load_k_vec4(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(k[index]);
#else
    return k[index];
#endif
}

fn load_v_vec4(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(v[index]);
#else
    return v[index];
#endif
}

fn store_output(index: u32, value: vec4<f32>) {
#ifdef FP16
    output[index] = pack4x16float(value);
#else
    output[index] = value;
#endif
}

// Symmetric power retention (deg=2) for Brumby / PowerCoder.
//
// Compile-time macros:
//   BLOCK_SIZE = HEAD_SIZE = head_dim / 4 (threads per workgroup)
//   HEAD_DIM = head_dim (128)
//   NUM_HEADS = number of Q heads (24)
//   NUM_KV_HEADS = number of KV heads (2)
//   GROUP_RATIO = NUM_HEADS / NUM_KV_HEADS (12)
//   EXPANDED_DIM = D = 9216 (symmetric power expanded dimension)
//   NUM_BLOCK_PAIRS = 72 (number of block-pairs in upper triangle)
//   NUM_GATE_HEADS, GATE_STRIDE = gate tensor layout
//
// Dispatch: [NUM_KV_HEADS, 1, 1] -- one workgroup per KV head.
// Each workgroup has BLOCK_SIZE threads, each handling one vec4 column (V-dimension).
//
// Feature map phi for deg=2 symmetric power:
//   K is split into outer blocks of 8 and inner blocks of 16.
//   phi_K[d] = mult * K[a(d)] * K[b(d)]  where a,b derived from block-pair structure.
//   phi_Q[d] = Q[a(d)] * Q[b(d)]  (no multiplier for Q).
//   Property: phi(Q) . phi(K) = (Q . K)^2
//
// Per token (sequential):
//   decay = sigmoid(gate[kv_head, token])
//   For each D-row d:
//     S[d, col] = decay * S[d, col] + phi_K[d] * V[col]
//     s[d] = decay * s[d] + phi_K[d]
//   For each Q head qg in [0, GROUP_RATIO):
//     Y[qg, col] = sum_d( S[d, col] * phi_Q_qg[d] )
//     l[qg] = sum_d( s[d] * phi_Q_qg[d] )
//     O[qg, col] = Y[qg, col] / l[qg]
//
@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn power_retention(in: Input) {
    let q_stride = NUM_HEADS * HEAD_SIZE;
    let kv_stride = NUM_KV_HEADS * HEAD_SIZE;
    let num_token = shape.y;

    let kv_head = in.wid.x;
    let idx = in.tid.x;

    // Sum-of-keys base offset for this KV head (flat indexing per batch).
    let sok_stride = NUM_KV_HEADS * EXPANDED_DIM;

    for (var t = 0u; t < num_token; t++) {
        let cursor = compute_cursor(cursors[t]);

        // Load K[head_dim] into shared_k (32 threads, each loads 4 f32 values).
        let k_vec4 = load_k_vec4(t * kv_stride + kv_head * HEAD_SIZE + idx);
        shared_k[idx * 4u + 0u] = k_vec4.x;
        shared_k[idx * 4u + 1u] = k_vec4.y;
        shared_k[idx * 4u + 2u] = k_vec4.z;
        shared_k[idx * 4u + 3u] = k_vec4.w;

        // Load Q[head_dim] for each Q head in the GQA group.
        for (var qg = 0u; qg < GROUP_RATIO; qg++) {
            let q_head = kv_head * GROUP_RATIO + qg;
            let q_vec4 = load_q_vec4(t * q_stride + q_head * HEAD_SIZE + idx);
            let qbase = qg * HEAD_DIM + idx * 4u;
            shared_q[qbase + 0u] = q_vec4.x;
            shared_q[qbase + 1u] = q_vec4.y;
            shared_q[qbase + 2u] = q_vec4.z;
            shared_q[qbase + 3u] = q_vec4.w;
        }
        workgroupBarrier();

        // Gate: one scalar per KV head per token.
        let gate_val = load_gate(t, kv_head);
        let decay = 1.0 / (1.0 + exp(-gate_val));

        // V column for this thread (vec4 of head_dim dimension).
        let v_col = load_v_vec4(t * kv_stride + kv_head * HEAD_SIZE + idx);

        // Output accumulators per Q head.
        var y0 = vec4<f32>(0.0);
        var y1 = vec4<f32>(0.0);
        var y2 = vec4<f32>(0.0);
        var y3 = vec4<f32>(0.0);
        var y4 = vec4<f32>(0.0);
        var y5 = vec4<f32>(0.0);
        var y6 = vec4<f32>(0.0);
        var y7 = vec4<f32>(0.0);
        var y8 = vec4<f32>(0.0);
        var y9 = vec4<f32>(0.0);
        var y10 = vec4<f32>(0.0);
        var y11 = vec4<f32>(0.0);

        // Normalizer accumulators (thread 0 only).
        var l0 = 0.0; var l1 = 0.0; var l2 = 0.0; var l3 = 0.0;
        var l4 = 0.0; var l5 = 0.0; var l6 = 0.0; var l7 = 0.0;
        var l8 = 0.0; var l9 = 0.0; var l10 = 0.0; var l11 = 0.0;

        let sok_base = cursor.batch * sok_stride + kv_head * EXPANDED_DIM;

        // Iterate over block-pairs in the upper triangle.
        // y_K: inner block index (0..7), x_K: outer block index (0..2*(y_K+1)-1).
        // Each block-pair produces 8*16=128 D-rows.
        var d = 0u;
        for (var y_K = 0u; y_K < 8u; y_K++) {
            let max_x = 2u * (y_K + 1u);
            for (var x_K = 0u; x_K < max_x; x_K++) {
                // Multiplier: 2 for off-diagonal (x_K < 2*y_K), 1 for diagonal.
                let mult = select(1.0, 2.0, x_K < 2u * y_K);

                for (var o = 0u; o < 8u; o++) {
                    let a = x_K * 8u + o;

                    for (var i = 0u; i < 16u; i++) {
                        let b = y_K * 16u + i;

                        // Feature map values.
                        let phi_k = mult * shared_k[a] * shared_k[b];

                        // State update: S[d, col] = decay * S[d, col] + phi_k * V[col]
                        let si = compute_index(cursor.batch, d, kv_head * HEAD_SIZE + idx);
                        var s_val = state[si];
                        s_val = decay * s_val + phi_k * v_col;
                        state[si] = s_val;

                        // Output accumulation per Q head.
                        // phi_Q[d] = Q[a] * Q[b] (no multiplier for Q).
                        var pq0 = shared_q[0u * HEAD_DIM + a] * shared_q[0u * HEAD_DIM + b];
                        y0 += s_val * pq0;
                        var pq1 = shared_q[1u * HEAD_DIM + a] * shared_q[1u * HEAD_DIM + b];
                        y1 += s_val * pq1;
                        var pq2 = shared_q[2u * HEAD_DIM + a] * shared_q[2u * HEAD_DIM + b];
                        y2 += s_val * pq2;
                        var pq3 = shared_q[3u * HEAD_DIM + a] * shared_q[3u * HEAD_DIM + b];
                        y3 += s_val * pq3;
                        var pq4 = shared_q[4u * HEAD_DIM + a] * shared_q[4u * HEAD_DIM + b];
                        y4 += s_val * pq4;
                        var pq5 = shared_q[5u * HEAD_DIM + a] * shared_q[5u * HEAD_DIM + b];
                        y5 += s_val * pq5;
                        var pq6 = shared_q[6u * HEAD_DIM + a] * shared_q[6u * HEAD_DIM + b];
                        y6 += s_val * pq6;
                        var pq7 = shared_q[7u * HEAD_DIM + a] * shared_q[7u * HEAD_DIM + b];
                        y7 += s_val * pq7;
                        var pq8 = shared_q[8u * HEAD_DIM + a] * shared_q[8u * HEAD_DIM + b];
                        y8 += s_val * pq8;
                        var pq9 = shared_q[9u * HEAD_DIM + a] * shared_q[9u * HEAD_DIM + b];
                        y9 += s_val * pq9;
                        var pq10 = shared_q[10u * HEAD_DIM + a] * shared_q[10u * HEAD_DIM + b];
                        y10 += s_val * pq10;
                        var pq11 = shared_q[11u * HEAD_DIM + a] * shared_q[11u * HEAD_DIM + b];
                        y11 += s_val * pq11;

                        // Sum-of-keys update and normalizer accumulation (thread 0 only).
                        if (idx == 0u) {
                            let sok_idx = sok_base + d;
                            var sok = sum_of_keys[sok_idx];
                            sok = decay * sok + phi_k;
                            sum_of_keys[sok_idx] = sok;

                            l0 += sok * pq0;
                            l1 += sok * pq1;
                            l2 += sok * pq2;
                            l3 += sok * pq3;
                            l4 += sok * pq4;
                            l5 += sok * pq5;
                            l6 += sok * pq6;
                            l7 += sok * pq7;
                            l8 += sok * pq8;
                            l9 += sok * pq9;
                            l10 += sok * pq10;
                            l11 += sok * pq11;
                        }

                        d++;
                    }
                }
            }
        }

        // Thread 0 writes normalizers to shared memory.
        if (idx == 0u) {
            shared_l[0] = l0; shared_l[1] = l1; shared_l[2] = l2; shared_l[3] = l3;
            shared_l[4] = l4; shared_l[5] = l5; shared_l[6] = l6; shared_l[7] = l7;
            shared_l[8] = l8; shared_l[9] = l9; shared_l[10] = l10; shared_l[11] = l11;
        }
        workgroupBarrier();

        // Store normalized output for each Q head.
        let inv_l0 = 1.0 / shared_l[0];
        let inv_l1 = 1.0 / shared_l[1];
        let inv_l2 = 1.0 / shared_l[2];
        let inv_l3 = 1.0 / shared_l[3];
        let inv_l4 = 1.0 / shared_l[4];
        let inv_l5 = 1.0 / shared_l[5];
        let inv_l6 = 1.0 / shared_l[6];
        let inv_l7 = 1.0 / shared_l[7];
        let inv_l8 = 1.0 / shared_l[8];
        let inv_l9 = 1.0 / shared_l[9];
        let inv_l10 = 1.0 / shared_l[10];
        let inv_l11 = 1.0 / shared_l[11];

        let q_base = kv_head * GROUP_RATIO;
        store_output(t * q_stride + (q_base + 0u) * HEAD_SIZE + idx, y0 * inv_l0);
        store_output(t * q_stride + (q_base + 1u) * HEAD_SIZE + idx, y1 * inv_l1);
        store_output(t * q_stride + (q_base + 2u) * HEAD_SIZE + idx, y2 * inv_l2);
        store_output(t * q_stride + (q_base + 3u) * HEAD_SIZE + idx, y3 * inv_l3);
        store_output(t * q_stride + (q_base + 4u) * HEAD_SIZE + idx, y4 * inv_l4);
        store_output(t * q_stride + (q_base + 5u) * HEAD_SIZE + idx, y5 * inv_l5);
        store_output(t * q_stride + (q_base + 6u) * HEAD_SIZE + idx, y6 * inv_l6);
        store_output(t * q_stride + (q_base + 7u) * HEAD_SIZE + idx, y7 * inv_l7);
        store_output(t * q_stride + (q_base + 8u) * HEAD_SIZE + idx, y8 * inv_l8);
        store_output(t * q_stride + (q_base + 9u) * HEAD_SIZE + idx, y9 * inv_l9);
        store_output(t * q_stride + (q_base + 10u) * HEAD_SIZE + idx, y10 * inv_l10);
        store_output(t * q_stride + (q_base + 11u) * HEAD_SIZE + idx, y11 * inv_l11);
        workgroupBarrier();
    }
}
