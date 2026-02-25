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
// view for state tensor: [num_emb, head_dim, batch, 1]
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

var<workgroup> shared_q: array<vec4<f32>, BLOCK_SIZE>;
var<workgroup> shared_v: array<vec4<f32>, BLOCK_SIZE>;

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
    // Map Q head to gate head (supports both per-head and per-KV-head gates).
    let gate_head = head * NUM_GATE_HEADS / NUM_HEADS;
    let flat = token * NUM_GATE_HEADS + gate_head;
    let vec_idx = flat / 4u;
    let component = flat % 4u;
#ifdef FP16
    let values = unpack4x16float(gate[vec_idx]);
#else
    let values = gate[vec_idx];
#endif
    return values[component];
}

fn load_q(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(q[index]);
#else
    return q[index];
#endif
}

fn load_k(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(k[index]);
#else
    return k[index];
#endif
}

fn load_v(index: u32) -> vec4<f32> {
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

// Power retention for Brumby.
//
// Compile-time macros: BLOCK_SIZE, HEAD_SIZE, NUM_HEADS, NUM_KV_HEADS, FP16.
// BLOCK_SIZE = HEAD_SIZE = head_dim / 4.
//
// Dispatch: [NUM_HEADS, 1, 1] — one workgroup per Q head.
// Each workgroup has BLOCK_SIZE threads, each handling one vec4 column of the state.
//
// For each token (sequential):
//   decay = exp(gate[head, token])
//   S[head] = decay * S[head] + outer(V[kv_head], K[kv_head])
//   Y[head] = S[head] @ Q[head]
//
@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn power_retention(in: Input) {
    let q_stride = NUM_HEADS * HEAD_SIZE;
    let kv_stride = NUM_KV_HEADS * HEAD_SIZE;
    let num_token = shape.y;

    let head = in.wid.x;
    let idx = in.tid.x;
    let kv_head = head * NUM_KV_HEADS / NUM_HEADS;

    for (var t = 0u; t < num_token; t++) {
        let cursor = compute_cursor(cursors[t]);

        // Load Q and V for this token into shared memory.
        shared_q[idx] = load_q(t * q_stride + head * HEAD_SIZE + idx);
        shared_v[idx] = load_v(t * kv_stride + kv_head * HEAD_SIZE + idx);
        workgroupBarrier();

        // Gate: one scalar per Q head per token.
        let decay = exp(load_gate(t, head));

        // K for our column position from the KV head.
        let k_vec = load_k(t * kv_stride + kv_head * HEAD_SIZE + idx);

        var y = vec4<f32>(0.0);

        // Iterate over rows of S in groups of 4 (matching vec4 of V and Q).
        for (var j = 0u; j < HEAD_SIZE; j++) {
            let v_scalar = shared_v[j];
            let q_scalar = shared_q[j];

            // State indices for 4 consecutive rows.
            let si0 = compute_index(cursor.batch, j * 4u + 0u, head * HEAD_SIZE + idx);
            let si1 = compute_index(cursor.batch, j * 4u + 1u, head * HEAD_SIZE + idx);
            let si2 = compute_index(cursor.batch, j * 4u + 2u, head * HEAD_SIZE + idx);
            let si3 = compute_index(cursor.batch, j * 4u + 3u, head * HEAD_SIZE + idx);

            var s0 = state[si0];
            var s1 = state[si1];
            var s2 = state[si2];
            var s3 = state[si3];

            // State update: S[row][col] = decay * S[row][col] + V[row] * K[col]
            s0 = decay * s0 + v_scalar[0] * k_vec;
            s1 = decay * s1 + v_scalar[1] * k_vec;
            s2 = decay * s2 + v_scalar[2] * k_vec;
            s3 = decay * s3 + v_scalar[3] * k_vec;

            state[si0] = s0;
            state[si1] = s1;
            state[si2] = s2;
            state[si3] = s3;

            // Output: Y[col] += S[row][col] * Q[row]
            y += s0 * q_scalar[0];
            y += s1 * q_scalar[1];
            y += s2 * q_scalar[2];
            y += s3 * q_scalar[3];
        }

        store_output(t * q_stride + head * HEAD_SIZE + idx, y);
        workgroupBarrier();
    }
}
