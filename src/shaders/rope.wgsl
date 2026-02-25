struct Input {
    @builtin(global_invocation_id) uid: vec3<u32>,
};

struct Cursor {
    batch: u32,
    token: u32,
    len: u32,
};

// shape comes from x.meta_layout: [num_heads * head_dim, num_token, 1, 1]
@group(0) @binding(0) var<uniform> shape: vec4<u32>;
@group(0) @binding(1) var<storage, read> cursors: array<u32>;

#ifdef FP16
@group(0) @binding(2) var<storage, read_write> x: array<vec2<u32>>;
#else
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
#endif

fn pack4x16float(x: vec4<f32>) -> vec2<u32> {
    return vec2<u32>(pack2x16float(x.xy), pack2x16float(x.zw));
}

fn unpack4x16float(x: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(unpack2x16float(x.x), unpack2x16float(x.y));
}

fn load_x(index: u32) -> vec4<f32> {
#ifdef FP16
    return unpack4x16float(x[index]);
#else
    return x[index];
#endif
}

fn store_x(index: u32, value: vec4<f32>) {
#ifdef FP16
    x[index] = pack4x16float(value);
#else
    x[index] = value;
#endif
}

fn compute_cursor(c: u32) -> Cursor {
    var cursor: Cursor;
    cursor.batch = c & 0xffu;
    cursor.token = (c >> 8u) & 0xffffu;
    cursor.len = (c >> 24u) & 0xffu;
    return cursor;
}

// HEAD_DIM and NUM_HEADS are compile-time macros.
// x has shape [NUM_HEADS * HEAD_DIM, num_token, 1, 1].
// We process in vec4 units, so head_dim_4 = HEAD_DIM / 4.
// Dispatch: [ceil(head_dim_4 / BLOCK_SIZE), NUM_HEADS, num_token]
@compute @workgroup_size(BLOCK_SIZE, 1, 1)
fn rope(in: Input) {
    let head_dim = HEAD_DIM;
    let head_dim_4 = head_dim / 4u;
    let num_heads = NUM_HEADS;
    let num_token = shape.y;    // from tensor meta: [total_dim, num_token, 1, 1]

    // Each thread handles one vec4 (two dimension pairs) for one head of one token.
    let vec_idx = in.uid.x;         // which vec4 within a head
    let head = in.uid.y;            // which head
    let token = in.uid.z;           // which token

    if vec_idx >= head_dim_4 || head >= num_heads || token >= num_token {
        return;
    }

    // Compute the absolute position from cursor encoding.
    let cursor = compute_cursor(cursors[token]);
    let pos = f32(cursor.token);

    // Index into the data array.
    // Layout: [num_heads * head_dim / 4, num_token] in vec4 units.
    let stride = head_dim_4 * num_heads;
    let index = token * stride + head * head_dim_4 + vec_idx;

    let val = load_x(index);

    // Dimension indices within the head for this vec4.
    let dim_base = vec_idx * 4u;    // base dimension index

    // Pair 0: dimensions (dim_base, dim_base + 1)
    let i0 = f32(dim_base) / f32(head_dim);
    let theta0 = pos * pow(ROPE_THETA, -i0);
    let cos0 = cos(theta0);
    let sin0 = sin(theta0);

    // Pair 1: dimensions (dim_base + 2, dim_base + 3)
    let i1 = f32(dim_base + 2u) / f32(head_dim);
    let theta1 = pos * pow(ROPE_THETA, -i1);
    let cos1 = cos(theta1);
    let sin1 = sin(theta1);

    // Apply rotation.
    let out = vec4<f32>(
        val.x * cos0 - val.y * sin0,
        val.x * sin0 + val.y * cos0,
        val.z * cos1 - val.w * sin1,
        val.z * sin1 + val.w * cos1,
    );

    store_x(index, out);
}
