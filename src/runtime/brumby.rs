use std::{collections::HashMap, marker::PhantomData, sync::Arc};

#[cfg(not(target_arch = "wasm32"))]
use futures::future::BoxFuture;
#[cfg(target_arch = "wasm32")]
use futures::future::LocalBoxFuture;
use half::f16;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use web_rwkv_derive::DeserializeSeed;
use wgpu::CommandBuffer;

use super::{
    infer::{RnnChunk, RnnInfo, RnnInput, RnnOutput, RnnOutputBatch, RnnRedirect, Token},
    loader::{Loader, LoaderError, Reader},
    model::{AsAny, ModelBuilder, ModelCustomInfo, ModelInfo, State as _},
    Dispatcher, Job, RuntimeError,
};
use crate::{
    context::Context,
    num::Float,
    runtime::model::Quant,
    tensor::{
        cache::ResourceCache,
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorCommand, TensorOp},
        serialization::Seed,
        shape::Shape,
        DeepClone, IntoPackedCursors, TensorCpu, TensorError, TensorGpu, TensorGpuView, TensorInit,
        TensorShape, TensorStack,
    },
};

/// Custom info for the Brumby model family (Brumby-14B, PowerCoder-3B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CustomInfo {
    /// Number of key-value heads (GQA).
    pub num_kv_head: usize,
    /// Dimension per attention head.
    pub head_dim: usize,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// Whether the model uses QK norms (true for Brumby, false for PowerCoder).
    pub has_qk_norm: bool,
    /// Whether the FFN is gated (true for Brumby, false for PowerCoder).
    pub gated_ffn: bool,
    /// Hidden activation function (Silu for Brumby, Gelu for PowerCoder).
    pub hidden_act: Activation,
    /// RoPE theta stored as f32 bits (1e6 for Brumby, 1e4 for PowerCoder).
    pub rope_theta_bits: u32,
}

impl CustomInfo {
    pub fn rope_theta(&self) -> f32 {
        f32::from_bits(self.rope_theta_bits)
    }

    pub fn with_rope_theta(mut self, theta: f32) -> Self {
        self.rope_theta_bits = theta.to_bits();
        self
    }
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Model {
    pub context: Context,
    pub info: ModelInfo,
    pub rescale: usize,
    pub sep: usize,
    pub tensor: ModelTensor,
}

impl Model {
    pub const RMS_EPS: f32 = 1.0e-6;

    pub const DEFAULT_RESCALE: usize = usize::MAX;
    pub const DEFAULT_SEP: usize = 1024;
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct ModelTensor {
    pub embed: Embed,
    pub head: Head,
    pub layers: Vec<Layer>,
}

/// RMS normalization (weight only, no bias).
/// Stores a pre-allocated zero bias for compatibility with the `rms_norm` shader.
#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct RmsNorm {
    pub w: TensorGpu<f16, ReadWrite>,
    pub b: TensorGpu<f16, ReadWrite>,
}

/// Power retention attention layer.
#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Att {
    pub q_proj: Matrix,
    pub k_proj: Matrix,
    pub v_proj: Matrix,
    pub o_proj: Matrix,
    pub g_proj: Matrix,

    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,

    pub q_bias: Option<TensorGpu<f16, ReadWrite>>,
    pub k_bias: Option<TensorGpu<f16, ReadWrite>>,
    pub v_bias: Option<TensorGpu<f16, ReadWrite>>,
    pub o_bias: Option<TensorGpu<f16, ReadWrite>>,
    pub g_bias: Option<TensorGpu<f16, ReadWrite>>,
}

/// Feed-forward network (gated or dense).
#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub enum Ffn {
    /// Brumby: gate_proj -> SiLU, up_proj, mul, down_proj
    Gated {
        gate_proj: Matrix,
        up_proj: Matrix,
        down_proj: Matrix,
    },
    /// PowerCoder: c_fc -> GELU, c_proj
    Dense {
        c_fc: Matrix,
        c_fc_bias: Option<TensorGpu<f16, ReadWrite>>,
        c_proj: Matrix,
        c_proj_bias: Option<TensorGpu<f16, ReadWrite>>,
    },
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Layer {
    pub input_ln: RmsNorm,
    pub post_att_ln: RmsNorm,
    pub att: Att,
    pub ffn: Ffn,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Embed {
    pub w: TensorCpu<f16>,
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Head {
    pub ln: RmsNorm,
    pub w: Matrix,
}

/// Power retention state.
///
/// For each layer, the state contains:
/// - The retention state matrix S: `[num_head, head_dim, head_dim]` (stored as f32)
/// - A token shift register: `[num_emb]`
///
/// All packed into a single tensor per layer.
#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct State {
    pub context: Context,
    pub info: ModelInfo,
    pub data: Vec<TensorGpu<f32, ReadWrite>>,
}

impl State {
    async fn back(&self, batch: usize) -> Result<TensorCpu<f32>, TensorError> {
        let context = &self.context;
        let mut tensors = Vec::with_capacity(self.info.num_layer);
        let mut encoder = context.device.create_command_encoder(&Default::default());
        for data in self.data.iter() {
            let shape = data.shape();
            let destination = context.tensor_init([shape[0], shape[1], 1, 1]);
            encoder.copy_tensor_batch(data, &destination, batch, 0)?;
            tensors.push(destination);
        }
        context.queue.submit(Some(encoder.finish()));

        let mut backed = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            backed.push(tensor.back().await);
        }
        TensorCpu::stack(backed, 2)
    }
}

impl AsAny for State {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl super::model::State for State {
    #[inline]
    fn num_batch(&self) -> usize {
        self.data[0].shape()[2]
    }

    #[inline]
    fn init_shape(&self) -> Shape {
        let info = &self.info;
        let head_dim = info.num_emb / info.num_head;
        // State per layer: head_dim rows for the S matrix + 1 row for token shift
        [info.num_emb, head_dim + 1, info.num_layer, 1].into()
    }

    fn init(&self) -> TensorCpu<f32> {
        let shape = self.init_shape();
        let data = vec![0.0; shape.len()];
        TensorCpu::from_data(shape, data).unwrap()
    }

    fn att(&self, layer: usize) -> Result<TensorGpuView<'_, f32>, TensorError> {
        let head_dim = self.info.num_emb / self.info.num_head;
        self.data[layer].view(.., 0..head_dim, .., ..)
    }

    fn ffn(&self, layer: usize) -> Result<TensorGpuView<'_, f32>, TensorError> {
        let head_dim = self.info.num_emb / self.info.num_head;
        self.data[layer].view(.., head_dim, .., ..)
    }

    fn load(&self, tensor: TensorCpu<f32>, batch: usize) -> Result<(), TensorError> {
        let head_dim = self.info.num_emb / self.info.num_head;
        tensor.check_shape([self.info.num_emb, head_dim + 1, self.info.num_layer, 1])?;
        for (data, source) in self.data.iter().zip(tensor.split(2)?.into_iter()) {
            data.load_batch(&source, batch)?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn back(&self, batch: usize) -> BoxFuture<'_, Result<TensorCpu<f32>, TensorError>> {
        Box::pin(self.back(batch))
    }

    #[cfg(target_arch = "wasm32")]
    fn back(&self, batch: usize) -> LocalBoxFuture<'_, Result<TensorCpu<f32>, TensorError>> {
        Box::pin(self.back(batch))
    }

    fn write(&self, tensor: TensorGpu<f32, ReadWrite>, batch: usize) -> Result<(), TensorError> {
        let head_dim = self.info.num_emb / self.info.num_head;
        tensor.check_shape([self.info.num_emb, head_dim + 1, self.info.num_layer, 1])?;

        let context = &self.context;
        let mut ops = Vec::with_capacity(self.data.len());
        for (layer, data) in self.data.iter().enumerate() {
            ops.push(TensorOp::blit(
                tensor.view(.., .., layer, ..)?,
                data.view(.., .., batch, ..)?,
            )?);
        }
        context.queue.submit(context.encode(&TensorOp::List(ops)));

        Ok(())
    }

    fn read(&self, batch: usize) -> Result<TensorGpu<f32, ReadWrite>, TensorError> {
        let context = &self.context;
        let head_dim = self.info.num_emb / self.info.num_head;
        let shape = [self.info.num_emb, head_dim + 1, self.info.num_layer, 1];
        let tensor: TensorGpu<_, _> = context.tensor_init(shape);

        let mut ops = Vec::with_capacity(self.data.len());
        for (layer, data) in self.data.iter().enumerate() {
            ops.push(TensorOp::blit(
                data.view(.., .., batch, ..)?,
                tensor.view(.., .., layer, ..)?,
            )?);
        }
        context.queue.submit(context.encode(&TensorOp::List(ops)));

        Ok(tensor)
    }

    fn embed(&self, layer: usize, backed: TensorCpu<f32>) -> Result<TensorCpu<f32>, TensorError> {
        backed.slice(.., 0, layer, ..)
    }
}

impl DeepClone for State {
    fn deep_clone(&self) -> Self {
        let data = self.data.iter().map(|tensor| tensor.deep_clone()).collect();
        Self {
            data,
            ..self.clone()
        }
    }
}

/// Runtime buffers for Brumby inference.
#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Runtime<F: Float> {
    pub cursors: TensorGpu<u32, ReadWrite>,
    pub input: TensorGpu<f16, ReadWrite>,

    pub x: TensorGpu<F, ReadWrite>,

    /// Attention intermediates.
    pub att_x: TensorGpu<F, ReadWrite>,
    pub att_q: TensorGpu<F, ReadWrite>,
    pub att_k: TensorGpu<F, ReadWrite>,
    pub att_v: TensorGpu<F, ReadWrite>,
    pub att_g: TensorGpu<F, ReadWrite>,
    pub att_o: TensorGpu<F, ReadWrite>,

    /// FFN intermediates.
    pub ffn_x: TensorGpu<F, ReadWrite>,
    pub ffn_gate: TensorGpu<F, ReadWrite>,
    pub ffn_up: TensorGpu<F, ReadWrite>,
    pub ffn_down: TensorGpu<F, ReadWrite>,
}

impl<F: Float> Runtime<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_token: usize) -> Self {
        let ModelCustomInfo::Brumby(custom) = info.custom else {
            unreachable!()
        };

        let shape = Shape::new(info.num_emb, num_token, 1, 1);
        let cursors_shape = Shape::new(num_token, 1, 1, 1);
        let kv_shape = Shape::new(custom.num_kv_head * custom.head_dim, num_token, 1, 1);
        let num_gate_head = if custom.has_qk_norm {
            info.num_head
        } else {
            custom.num_kv_head
        };
        let gate_shape = Shape::new(num_gate_head, num_token, 1, 1);
        let ffn_hidden_shape = Shape::new(custom.intermediate_size, num_token, 1, 1);

        Self {
            cursors: context.tensor_init(cursors_shape),
            input: context.tensor_init(shape),
            x: context.tensor_init(shape),
            att_x: context.tensor_init(shape),
            att_q: context.tensor_init(shape),
            att_k: context.tensor_init(kv_shape),
            att_v: context.tensor_init(kv_shape),
            att_g: context.tensor_init(gate_shape),
            att_o: context.tensor_init(shape),
            ffn_x: context.tensor_init(shape),
            ffn_gate: context.tensor_init(ffn_hidden_shape),
            ffn_up: context.tensor_init(ffn_hidden_shape),
            ffn_down: context.tensor_init(shape),
        }
    }
}

#[derive(Debug, Clone, Serialize, DeserializeSeed)]
#[serde_seed(seed = "Seed", context = "Context")]
pub struct Header<F: Float> {
    pub head_x: TensorGpu<F, ReadWrite>,
    pub head_o: TensorGpu<f32, ReadWrite>,
}

impl<F: Float> Header<F> {
    pub fn new(context: &Context, info: &ModelInfo, num_header: usize) -> Self {
        let head_shape = Shape::new(info.num_emb, num_header, 1, 1);
        let output_shape = Shape::new(info.num_vocab_padded(), num_header, 1, 1);

        Self {
            head_x: context.tensor_init(head_shape),
            head_o: context.tensor_init(output_shape),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Hook {
    PostEmbedLoaded,
    PreAtt(usize),
    PostAttLayerNorm(usize),
    PreAttLinear(usize),
    PostAttLinear(usize),
    PreAttRetention(usize),
    PostAttRetention(usize),
    PreAttOut(usize),
    PostAttOut(usize),
    PostAtt(usize),
    PreFfn(usize),
    PostFfnLayerNorm(usize),
    PreFfnLinear(usize),
    PostFfnGate(usize),
    PostFfnLinear(usize),
    PostFfn(usize),
    PreHead,
    PostHeadLayerNorm,
    PostHead,
}

pub struct RnnJob {
    commands: Vec<CommandBuffer>,
    redirect: RnnRedirect,

    embed: TensorCpu<f16>,

    cursors: TensorGpu<u32, ReadWrite>,
    input: TensorGpu<f16, ReadWrite>,
    output: TensorGpu<f32, ReadWrite>,
}

impl Job for RnnJob {
    type Input = RnnInput;
    type Output = RnnOutput;

    fn load(&self, input: &RnnChunk) -> Result<(), RuntimeError> {
        if input.num_token() == 0 {
            return Ok(());
        }

        let stack: Vec<TensorCpu<f16>> = input
            .iter()
            .map(|chunk| {
                let num_emb = self.embed.shape()[0];
                let data = self.embed.data();
                let data = chunk
                    .iter()
                    .map(|token| match token {
                        &Token::Token(token) => {
                            let start = num_emb * token as usize;
                            let end = start + num_emb;
                            let data = data[start..end].to_vec();
                            TensorCpu::from_data_1d(data)
                        }
                        Token::Embed(tensor) => tensor.clone(),
                    })
                    .collect_vec();
                match TensorCpu::stack(data, 1) {
                    Ok(tensor) => tensor,
                    Err(_) => TensorCpu::init([num_emb, 0, 1, 1]),
                }
            })
            .collect();
        let stack = TensorStack::try_from(stack)?;

        let cursors = stack.cursors.clone().into_cursors();
        let cursors = TensorCpu::from_data(self.cursors.shape(), cursors)?;
        self.cursors.load(&cursors)?;
        self.input.load(&stack.tensor)?;

        Ok(())
    }

    fn submit(&mut self) {
        let commands = std::mem::take(&mut self.commands);
        self.output.context.queue.submit(commands);
    }

    async fn back(self) -> Result<Self::Output, RuntimeError> {
        let output = self.output.back().await;
        let batches: Vec<_> = self
            .redirect
            .outputs
            .into_iter()
            .map(|(start, end)| output.slice(.., start..end, .., ..))
            .try_collect()?;
        let batches = batches.into_iter().map(RnnOutputBatch).collect();
        Ok(RnnOutput(batches))
    }
}

#[derive(Debug, Clone)]
pub struct Frame<F: Float> {
    pub state: State,
    pub buffer: Arc<Runtime<F>>,
    pub header: Arc<Header<F>>,
}

pub type HookFn<F> = Box<dyn Fn(Frame<F>) -> Result<TensorOp, TensorError> + Send + Sync>;
pub type HookMap<F> = HashMap<Hook, HookFn<F>>;

#[derive(Clone)]
pub struct Bundle<F: Float> {
    model: Model,
    state: State,
    hooks: Arc<HookMap<F>>,
    buffers: ResourceCache<usize, Runtime<F>>,
    headers: ResourceCache<usize, Header<F>>,
    phantom: PhantomData<F>,
}

impl<F: Float> Bundle<F> {
    pub fn new(model: Model, num_batch: usize) -> Self {
        let context = model.context.clone();
        let info = model.info.clone();
        let state = {
            let head_dim = info.num_emb / info.num_head;
            let shape = Shape::new(info.num_emb, head_dim + 1, num_batch, 1);
            let data = (0..info.num_layer).map(|_| context.zeros(shape)).collect();
            State {
                context,
                info,
                data,
            }
        };
        Self {
            model,
            state,
            hooks: Default::default(),
            buffers: ResourceCache::new(4),
            headers: ResourceCache::new(4),
            phantom: PhantomData,
        }
    }

    pub fn new_with_hooks(model: Model, num_batch: usize, hooks: HookMap<F>) -> Self {
        Self {
            hooks: Arc::new(hooks),
            ..Self::new(model, num_batch)
        }
    }

    fn checkout_buffer(
        &self,
        context: &Context,
        info: &ModelInfo,
        num_token: usize,
    ) -> Arc<Runtime<F>> {
        self.buffers
            .checkout(num_token, || Runtime::new(context, info, num_token))
    }

    fn checkout_header(
        &self,
        context: &Context,
        info: &ModelInfo,
        num_header: usize,
    ) -> Arc<Header<F>> {
        self.headers
            .checkout(num_header, || Header::new(context, info, num_header))
    }
}

impl<F: Float> super::model::Bundle for Bundle<F> {
    #[inline]
    fn info(&self) -> ModelInfo {
        self.model.info.clone()
    }

    #[inline]
    fn state(&self) -> impl super::model::State + AsAny + 'static {
        self.state.clone()
    }

    #[inline]
    fn model(&self) -> impl Serialize + 'static {
        self.model.clone()
    }
}

fn turbo(num_token: usize) -> bool {
    num_token.is_multiple_of(super::infer::rnn::MIN_TOKEN_CHUNK_SIZE)
}

fn hook_op<F: Float>(
    hooks: &HookMap<F>,
    hook: &Hook,
    frame: &Frame<F>,
) -> Result<TensorOp, TensorError> {
    match hooks.get(hook) {
        Some(f) => f(frame.clone()),
        None => Ok(TensorOp::empty()),
    }
}

impl<F: Float> Dispatcher<RnnJob> for Bundle<F> {
    type Info = RnnInfo;

    fn dispatch(&self, seed: Self::Info) -> Result<RnnJob, RuntimeError> {
        let model = &self.model;
        let state = &self.state;
        let context = &model.context;
        let info = &model.info;
        let tensor = &model.tensor;

        let num_token = seed.num_token();
        let head_dim = info.num_emb / info.num_head;

        let redirect = seed.redirect();
        let num_header = redirect.headers.len();

        let buffer = self.checkout_buffer(context, info, num_token);
        let header = self.checkout_header(context, info, num_header);
        let frame = Frame {
            state: state.clone(),
            buffer: buffer.clone(),
            header: header.clone(),
        };

        context.maintain();
        self.buffers.maintain();
        self.headers.maintain();

        if num_token == 0 {
            return Ok(RnnJob {
                commands: vec![],
                redirect,
                embed: model.tensor.embed.w.clone(),
                cursors: buffer.cursors.clone(),
                input: buffer.input.clone(),
                output: header.head_o.clone(),
            });
        }

        #[cfg(feature = "trace")]
        let _span = tracing::trace_span!("build").entered();

        let (head_op, head_x) = redirect.op(&buffer.x, &header.head_x)?;

        let hook_op = |hook: Hook| hook_op(&self.hooks, &hook, &frame);
        let mut ops = vec![];

        // Brumby has no embedding layer norm - just load embeddings directly.
        {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("embed").entered();

            ops.extend([
                hook_op(Hook::PostEmbedLoaded)?,
                TensorOp::blit(&buffer.input, &buffer.x)?,
            ]);
        };

        for (index, layer) in tensor.layers.iter().enumerate() {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("layer", index).entered();

            let hooks = self.hooks.clone();
            let frame = frame.clone();
            let layer = layer.clone();

            let op = dispatch_layer(
                hooks,
                frame,
                layer,
                index,
                num_token,
                head_dim,
                model.rescale,
                info,
            )?;
            ops.push(op);

            if (index + 1) % model.sep == 0 {
                ops.push(TensorOp::Sep);
            }
        }

        {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("header").entered();

            let hooks = self.hooks.clone();
            let frame = frame.clone();
            let head = model.tensor.head.clone();

            let op = dispatch_header(hooks, frame, head, head_x, num_header, head_op)?;
            ops.push(op);
        }

        let commands = {
            #[cfg(feature = "trace")]
            let _span = tracing::trace_span!("encode").entered();
            context.encode(&TensorOp::List(ops))
        };

        Ok(RnnJob {
            commands,
            redirect,
            embed: model.tensor.embed.w.clone(),
            cursors: buffer.cursors.clone(),
            input: buffer.input.clone(),
            output: header.head_o.clone(),
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_layer<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    layer: Layer,
    index: usize,
    num_token: usize,
    _head_dim: usize,
    rescale: usize,
    info: &ModelInfo,
) -> Result<TensorOp, TensorError> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let Frame { state, buffer, .. } = &frame;
    let ModelCustomInfo::Brumby(custom) = info.custom else {
        unreachable!()
    };
    let num_gate_head = if custom.has_qk_norm {
        info.num_head
    } else {
        custom.num_kv_head
    };

    let mut ops = vec![];

    // === Attention block ===
    ops.extend([
        TensorOp::blit(&buffer.x, &buffer.att_x)?,
        hook_op(Hook::PreAtt(index))?,
    ]);

    // Input RMS norm
    ops.push(TensorOp::rms_norm(
        &layer.input_ln.w,
        &layer.input_ln.b,
        &buffer.att_x,
        Model::RMS_EPS,
    )?);
    ops.push(hook_op(Hook::PostAttLayerNorm(index))?);

    // Q, K, V, G projections
    ops.push(hook_op(Hook::PreAttLinear(index))?);
    ops.push(layer.att.q_proj.matmul_op(
        &buffer.att_x,
        &buffer.att_q,
        Activation::None,
        turbo(num_token),
    )?);
    if let Some(ref bias) = layer.att.q_bias {
        ops.push(TensorOp::add(bias, &buffer.att_q)?);
    }
    ops.push(layer.att.k_proj.matmul_op(
        &buffer.att_x,
        &buffer.att_k,
        Activation::None,
        turbo(num_token),
    )?);
    if let Some(ref bias) = layer.att.k_bias {
        ops.push(TensorOp::add(bias, &buffer.att_k)?);
    }
    ops.push(layer.att.v_proj.matmul_op(
        &buffer.att_x,
        &buffer.att_v,
        Activation::None,
        turbo(num_token),
    )?);
    if let Some(ref bias) = layer.att.v_bias {
        ops.push(TensorOp::add(bias, &buffer.att_v)?);
    }
    ops.push(layer.att.g_proj.matmul_op(
        &buffer.att_x,
        &buffer.att_g,
        Activation::None,
        turbo(num_token),
    )?);
    if let Some(ref bias) = layer.att.g_bias {
        ops.push(TensorOp::add(bias, &buffer.att_g)?);
    }
    ops.push(hook_op(Hook::PostAttLinear(index))?);

    // Post-projection RMS norms on Q and K (only for models with QK norms)
    if let Some(ref q_norm) = layer.att.q_norm {
        ops.push(TensorOp::group_rms_norm(
            &q_norm.w,
            &q_norm.b,
            &buffer.att_q,
            custom.head_dim as u32,
            Model::RMS_EPS,
        )?);
    }
    if let Some(ref k_norm) = layer.att.k_norm {
        ops.push(TensorOp::group_rms_norm(
            &k_norm.w,
            &k_norm.b,
            &buffer.att_k,
            custom.head_dim as u32,
            Model::RMS_EPS,
        )?);
    }

    // RoPE on Q and K
    let rope_theta = custom.rope_theta();
    ops.push(TensorOp::rope(
        &buffer.cursors,
        &buffer.att_q,
        custom.head_dim as u32,
        info.num_head as u32,
        rope_theta,
    )?);
    ops.push(TensorOp::rope(
        &buffer.cursors,
        &buffer.att_k,
        custom.head_dim as u32,
        custom.num_kv_head as u32,
        rope_theta,
    )?);

    // Power retention
    ops.push(hook_op(Hook::PreAttRetention(index))?);
    ops.push(TensorOp::power_retention(
        &buffer.cursors,
        state.att(index)?,
        &buffer.att_g,
        &buffer.att_q,
        &buffer.att_k,
        &buffer.att_v,
        &buffer.att_o,
        custom.head_dim as u32,
        info.num_head as u32,
        custom.num_kv_head as u32,
        num_gate_head as u32,
    )?);
    ops.push(hook_op(Hook::PostAttRetention(index))?);

    // Output projection
    ops.push(hook_op(Hook::PreAttOut(index))?);
    ops.push(layer.att.o_proj.matmul_op(
        &buffer.att_o,
        &buffer.att_x,
        Activation::None,
        turbo(num_token),
    )?);
    if let Some(ref bias) = layer.att.o_bias {
        ops.push(TensorOp::add(bias, &buffer.att_x)?);
    }
    ops.push(hook_op(Hook::PostAttOut(index))?);

    // Residual connection
    ops.push(TensorOp::add(&buffer.att_x, &buffer.x)?);
    ops.push(hook_op(Hook::PostAtt(index))?);

    // === FFN block ===
    ops.extend([
        TensorOp::blit(&buffer.x, &buffer.ffn_x)?,
        hook_op(Hook::PreFfn(index))?,
    ]);

    // Post-attention RMS norm
    ops.push(TensorOp::rms_norm(
        &layer.post_att_ln.w,
        &layer.post_att_ln.b,
        &buffer.ffn_x,
        Model::RMS_EPS,
    )?);
    ops.push(hook_op(Hook::PostFfnLayerNorm(index))?);

    ops.push(hook_op(Hook::PreFfnLinear(index))?);
    match &layer.ffn {
        Ffn::Gated {
            gate_proj,
            up_proj,
            down_proj,
        } => {
            // Gated SiLU FFN: output = down_proj(silu(gate_proj(x)) * up_proj(x))
            ops.push(gate_proj.matmul_op(
                &buffer.ffn_x,
                &buffer.ffn_gate,
                Activation::Silu,
                turbo(num_token),
            )?);
            ops.push(up_proj.matmul_op(
                &buffer.ffn_x,
                &buffer.ffn_up,
                Activation::None,
                turbo(num_token),
            )?);
            ops.push(hook_op(Hook::PostFfnGate(index))?);

            // Element-wise multiply: gate * up
            ops.push(TensorOp::mul(&buffer.ffn_gate, &buffer.ffn_up)?);
            // down_proj on the result
            ops.push(down_proj.matmul_op(
                &buffer.ffn_up,
                &buffer.ffn_down,
                Activation::None,
                turbo(num_token),
            )?);
        }
        Ffn::Dense {
            c_fc,
            c_fc_bias,
            c_proj,
            c_proj_bias,
        } => {
            // Dense FFN: output = c_proj(gelu(c_fc(x)))
            ops.push(c_fc.matmul_op(
                &buffer.ffn_x,
                &buffer.ffn_gate,
                Activation::None,
                turbo(num_token),
            )?);
            if let Some(ref bias) = c_fc_bias {
                ops.push(TensorOp::add(bias, &buffer.ffn_gate)?);
            }
            ops.push(TensorOp::activate(&buffer.ffn_gate, custom.hidden_act)?);
            ops.push(hook_op(Hook::PostFfnGate(index))?);

            ops.push(c_proj.matmul_op(
                &buffer.ffn_gate,
                &buffer.ffn_down,
                Activation::None,
                turbo(num_token),
            )?);
            if let Some(ref bias) = c_proj_bias {
                ops.push(TensorOp::add(bias, &buffer.ffn_down)?);
            }
        }
    }
    ops.push(hook_op(Hook::PostFfnLinear(index))?);

    // Residual connection
    ops.push(TensorOp::add(&buffer.ffn_down, &buffer.x)?);
    ops.push(hook_op(Hook::PostFfn(index))?);

    if (index + 1).is_multiple_of(rescale) {
        ops.push(TensorOp::affine(&buffer.x, 0.5, 0.0)?);
    }

    Ok(TensorOp::List(ops))
}

fn dispatch_header<F: Float>(
    hooks: Arc<HookMap<F>>,
    frame: Frame<F>,
    head: Head,
    head_x: TensorGpu<F, ReadWrite>,
    num_header: usize,
    head_op: TensorOp,
) -> Result<TensorOp, TensorError> {
    let hook_op = |hook: Hook| hook_op(&hooks, &hook, &frame);
    let header = &frame.header;
    let mut ops = vec![head_op];

    if num_header > 0 {
        ops.extend([
            hook_op(Hook::PreHead)?,
            TensorOp::rms_norm(&head.ln.w, &head.ln.b, &head_x, Model::RMS_EPS)?,
            hook_op(Hook::PostHeadLayerNorm)?,
            head.w.matmul_op(
                head_x.view(.., .., .., ..)?,
                header.head_o.view(.., .., .., ..)?,
                Activation::None,
                turbo(num_header),
            )?,
            hook_op(Hook::PostHead)?,
        ]);
    }
    Ok(TensorOp::List(ops))
}

impl<R: Reader> ModelBuilder<R> {
    pub async fn build_brumby(self) -> Result<Model, LoaderError> {
        let ModelBuilder {
            context,
            model,
            rescale,
            sep,
            lora,
            quant,
            ..
        } = self;

        let rescale = rescale.unwrap_or(Model::DEFAULT_RESCALE);
        let sep = sep.unwrap_or(Model::DEFAULT_SEP);

        let info = Loader::info(&model)?;
        let loader = Loader {
            context: context.clone(),
            model,
            lora,
        };

        let ModelCustomInfo::Brumby(custom) = info.custom else {
            unreachable!()
        };
        let use_bias = !custom.has_qk_norm; // PowerCoder has biases, Brumby does not

        /// Load an RMS norm: weight from file, bias is zero.
        fn load_rms_norm<R: Reader>(
            loader: &Loader<R>,
            context: &Context,
            name: impl AsRef<str>,
        ) -> Result<RmsNorm, LoaderError> {
            let w = loader.load_vector_f16(name)?;
            let b = context.zeros(w.shape());
            Ok(RmsNorm { w, b })
        }

        /// Load an RMS norm with real bias from file.
        fn load_rms_norm_with_bias<R: Reader>(
            loader: &Loader<R>,
            name: impl AsRef<str>,
        ) -> Result<RmsNorm, LoaderError> {
            let name = name.as_ref();
            let w = loader.load_vector_f16(format!("{name}.weight"))?;
            let b = loader.load_vector_f16(format!("{name}.bias"))?;
            Ok(RmsNorm { w, b })
        }

        let embed = Embed {
            w: loader.load_matrix_f16_padded_cpu("model.embed_tokens.weight")?,
        };

        let head = if use_bias {
            Head {
                ln: load_rms_norm_with_bias(&loader, "model.norm")?,
                w: Matrix::Fp16(loader.load_matrix_f16_padded("lm_head.weight")?),
            }
        } else {
            Head {
                ln: load_rms_norm(&loader, &context, "model.norm.weight")?,
                w: Matrix::Fp16(loader.load_matrix_f16_padded("lm_head.weight")?),
            }
        };

        let submission_index = Some(context.queue.submit(None));
        _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });

        let load_matrix = |name: String, quant: Quant| loader.load_matrix(name, quant);

        let mut layers = vec![];
        for layer in 0..info.num_layer {
            let quant = quant.get(&layer).copied().unwrap_or_default();

            let l = format!("model.layers.{layer}");

            let input_ln = if use_bias {
                load_rms_norm_with_bias(&loader, format!("{l}.input_layernorm"))?
            } else {
                load_rms_norm(&loader, &context, format!("{l}.input_layernorm.weight"))?
            };

            let load_bias = |name: String| -> Result<Option<TensorGpu<f16, ReadWrite>>, LoaderError> {
                if use_bias {
                    Ok(Some(loader.load_vector_f16(name)?))
                } else {
                    Ok(None)
                }
            };

            let att = Att {
                q_proj: load_matrix(format!("{l}.self_attn.q_proj.weight"), quant)?,
                k_proj: load_matrix(format!("{l}.self_attn.k_proj.weight"), quant)?,
                v_proj: load_matrix(format!("{l}.self_attn.v_proj.weight"), quant)?,
                o_proj: load_matrix(format!("{l}.self_attn.o_proj.weight"), quant)?,
                g_proj: load_matrix(format!("{l}.self_attn.g_proj.weight"), quant)?,
                q_norm: if custom.has_qk_norm {
                    Some(load_rms_norm(
                        &loader,
                        &context,
                        format!("{l}.self_attn.q_norm.weight"),
                    )?)
                } else {
                    None
                },
                k_norm: if custom.has_qk_norm {
                    Some(load_rms_norm(
                        &loader,
                        &context,
                        format!("{l}.self_attn.k_norm.weight"),
                    )?)
                } else {
                    None
                },
                q_bias: load_bias(format!("{l}.self_attn.q_proj.bias"))?,
                k_bias: load_bias(format!("{l}.self_attn.k_proj.bias"))?,
                v_bias: load_bias(format!("{l}.self_attn.v_proj.bias"))?,
                o_bias: load_bias(format!("{l}.self_attn.o_proj.bias"))?,
                g_bias: load_bias(format!("{l}.self_attn.g_proj.bias"))?,
            };

            let post_att_ln = if use_bias {
                load_rms_norm_with_bias(&loader, format!("{l}.post_attention_layernorm"))?
            } else {
                load_rms_norm(
                    &loader,
                    &context,
                    format!("{l}.post_attention_layernorm.weight"),
                )?
            };

            let ffn = if custom.gated_ffn {
                Ffn::Gated {
                    gate_proj: load_matrix(format!("{l}.mlp.gate_proj.weight"), quant)?,
                    up_proj: load_matrix(format!("{l}.mlp.up_proj.weight"), quant)?,
                    down_proj: load_matrix(format!("{l}.mlp.down_proj.weight"), quant)?,
                }
            } else {
                Ffn::Dense {
                    c_fc: load_matrix(format!("{l}.mlp.c_fc.weight"), quant)?,
                    c_fc_bias: load_bias(format!("{l}.mlp.c_fc.bias"))?,
                    c_proj: load_matrix(format!("{l}.mlp.c_proj.weight"), quant)?,
                    c_proj_bias: load_bias(format!("{l}.mlp.c_proj.bias"))?,
                }
            };

            let submission_index = Some(context.queue.submit(None));
            _ = context.device.poll(wgpu::PollType::Wait {
                submission_index,
                timeout: None,
            });

            layers.push(Layer {
                input_ln,
                post_att_ln,
                att,
                ffn,
            })
        }

        let submission_index = Some(context.queue.submit(None));
        _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });

        let tensor = ModelTensor {
            embed,
            head,
            layers,
        };
        let model = {
            let context = context.clone();
            let info = info.clone();
            Model {
                context,
                info,
                rescale,
                sep,
                tensor,
            }
        };
        Ok(model)
    }
}
