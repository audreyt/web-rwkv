use std::borrow::Cow;
use std::collections::HashMap;

use half::f16;
use safetensors::SafeTensorError;
use wasm_bindgen::prelude::*;

use crate::{
    context::{Context, ContextBuilder, InstanceExt as _},
    runtime::{
        brumby::{self, Att, Embed, Ffn, Head, Layer, Model, ModelTensor, RmsNorm},
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption, Token},
        loader::{Loader, Reader, ReaderTensor, TensorFromReader as _, PAD_MAT},
        model::{AsAny, Bundle as _, ContextAutoLimits as _, ModelBuilder, ModelCustomInfo, ModelInfo, Quant, State as _},
        softmax, SimpleRuntime,
    },
    tensor::{kind::ReadWrite, matrix::Matrix, TensorCpu, TensorGpu, TensorInit as _, TensorInto as _, TensorShape as _},
};

type BrumbyBundle = brumby::Bundle<f16>;
type BrumbyRuntime = SimpleRuntime<BrumbyBundle, Rnn, brumby::RnnJob>;

// ── MetadataReader ─────────────────────────────────────────
// A Reader backed only by tensor names and shapes (no data).
// Used to compute ModelInfo without loading any tensor bytes.

struct MetadataReader {
    all_names: Vec<String>,
    shapes: HashMap<String, Vec<usize>>,
}

impl Reader for MetadataReader {
    fn names(&self) -> Vec<&str> {
        self.all_names.iter().map(|s| s.as_str()).collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.all_names.iter().any(|n| n == name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        self.shapes
            .get(name)
            .cloned()
            .ok_or(SafeTensorError::TensorNotFound(name.to_string()))
    }

    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        Err(SafeTensorError::TensorNotFound(format!(
            "metadata-only reader: {name}"
        )))
    }
}

// ── WasmSessionBuilder (streaming) ─────────────────────────
// Uploads tensors to GPU during add_shard(), keeping peak WASM
// memory at ~1 shard (~4 GB) instead of all shards (~12 GB).

#[wasm_bindgen]
pub struct WasmSessionBuilder {
    /// Tensor name → shard filename, from the index JSON.
    #[wasm_bindgen(skip)]
    pub weight_map: HashMap<String, String>,

    token_chunk_size: u32,
    quant_layers: u32,

    /// GPU context, created in prepare().
    #[wasm_bindgen(skip)]
    pub context: Option<Context>,

    /// Accumulated tensor shapes from parsed shard headers.
    #[wasm_bindgen(skip)]
    pub tensor_meta: HashMap<String, Vec<usize>>,

    /// Pre-uploaded vector tensors (norms, biases) on GPU.
    #[wasm_bindgen(skip)]
    pub gpu_vectors: HashMap<String, TensorGpu<f16, ReadWrite>>,

    /// Pre-uploaded weight matrices on GPU (possibly quantized).
    #[wasm_bindgen(skip)]
    pub matrices: HashMap<String, Matrix>,

    /// Embedding weight — stays on CPU (padded).
    #[wasm_bindgen(skip)]
    pub embed_cpu: Option<TensorCpu<f16>>,
}

#[wasm_bindgen]
impl WasmSessionBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        index_json: &str,
        token_chunk_size: u32,
        quant_layers: u32,
    ) -> Result<WasmSessionBuilder, JsError> {
        #[derive(serde::Deserialize)]
        struct IndexJson {
            weight_map: HashMap<String, String>,
        }

        let parsed: IndexJson = serde_json::from_str(index_json)
            .map_err(|e| JsError::new(&format!("Invalid index JSON: {e}")))?;

        Ok(WasmSessionBuilder {
            weight_map: parsed.weight_map,
            token_chunk_size,
            quant_layers,
            context: None,
            tensor_meta: HashMap::new(),
            gpu_vectors: HashMap::new(),
            matrices: HashMap::new(),
            embed_cpu: None,
        })
    }

    /// Create the WebGPU context. Must be called before add_shard().
    pub async fn prepare(&mut self) -> Result<(), JsError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await
            .map_err(|e| JsError::new(&format!("WebGPU adapter error: {e}")))?;

        let mut ctx_builder = ContextBuilder::new(adapter);
        // Use generous limits; the adapter will clamp to what it supports.
        ctx_builder.limits.max_buffer_size = 1 << 30; // 1 GB
        ctx_builder.limits.max_storage_buffer_binding_size = 1 << 30;

        let context = ctx_builder
            .build()
            .await
            .map_err(|e| JsError::new(&format!("Context error: {e}")))?;

        self.context = Some(context);
        Ok(())
    }

    /// Parse one shard and upload all its tensors to GPU.
    ///
    /// The shard's bytes live in WASM memory only for the duration of this
    /// call — after it returns, the staging buffer is freed by the JS glue.
    ///
    /// NOTE: For shards > ~4 GB, use `add_tensor()` + `flush()` instead,
    /// because browsers limit a single JS ArrayBuffer to ~4 GB.
    pub fn add_shard(&mut self, name: &str, data: &[u8]) -> Result<(), JsError> {
        // Clone the context to avoid borrow conflict (Context is Arc-based, clone is cheap).
        let context = self
            .context
            .clone()
            .ok_or_else(|| JsError::new("call prepare() before add_shard()"))?;

        let st = safetensors::SafeTensors::deserialize(data)
            .map_err(|e| JsError::new(&format!("SafeTensor parse error: {e}")))?;

        // Collect tensor names that belong to this shard.
        let shard_tensors: Vec<String> = self
            .weight_map
            .iter()
            .filter(|(_, shard_file)| shard_file.as_str() == name)
            .map(|(tensor_name, _)| tensor_name.clone())
            .collect();

        for tensor_name in &shard_tensors {
            let tv = st.tensor(tensor_name).map_err(|e| {
                JsError::new(&format!("Missing tensor {tensor_name} in shard {name}: {e}"))
            })?;

            // Accumulate metadata for later ModelInfo computation.
            self.tensor_meta
                .insert(tensor_name.clone(), tv.shape().to_vec());

            let reader_tensor: ReaderTensor<'_> =
                (tv.dtype(), tv.shape().to_vec(), Cow::Borrowed(tv.data()));

            self.upload_tensor(tensor_name, reader_tensor, &context)?;
        }

        // Flush GPU uploads.
        let submission_index = Some(context.queue.submit(None));
        _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });

        Ok(())
    }

    /// Add a single tensor by name, shape, dtype string, and raw bytes.
    ///
    /// Call this from JS after parsing the safetensors header on the JS side.
    /// This avoids loading an entire shard into one ArrayBuffer (which fails
    /// for shards > ~4 GB due to browser limits).
    ///
    /// `dtype` must be a safetensors dtype string: "F16", "F32", "BF16", etc.
    pub fn add_tensor(
        &mut self,
        name: &str,
        shape: &[u32],
        dtype: &str,
        data: &[u8],
    ) -> Result<(), JsError> {
        let context = self
            .context
            .clone()
            .ok_or_else(|| JsError::new("call prepare() before add_tensor()"))?;

        let st_dtype = match dtype {
            "F16" => safetensors::Dtype::F16,
            "F32" => safetensors::Dtype::F32,
            "BF16" => safetensors::Dtype::BF16,
            "I32" => safetensors::Dtype::I32,
            "I64" => safetensors::Dtype::I64,
            "U8" => safetensors::Dtype::U8,
            "U16" => safetensors::Dtype::U16,
            "U32" => safetensors::Dtype::U32,
            "I8" => safetensors::Dtype::I8,
            "I16" => safetensors::Dtype::I16,
            _ => return Err(JsError::new(&format!("Unknown dtype: {dtype}"))),
        };

        let shape_usize: Vec<usize> = shape.iter().map(|&s| s as usize).collect();

        // Record metadata for later ModelInfo computation.
        self.tensor_meta
            .insert(name.to_string(), shape_usize.clone());

        // Upload to GPU (or keep on CPU for embed).
        let reader_tensor: ReaderTensor<'_> = (st_dtype, shape_usize, Cow::Borrowed(data));
        self.upload_tensor(name, reader_tensor, &context)?;

        Ok(())
    }

    /// Flush pending GPU uploads. Call after a batch of add_tensor() calls
    /// (typically once per shard).
    pub fn flush(&self) -> Result<(), JsError> {
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| JsError::new("call prepare() before flush()"))?;
        let submission_index = Some(context.queue.submit(None));
        _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });
        Ok(())
    }

    /// Assemble the model from pre-uploaded GPU tensors.
    pub async fn build(mut self) -> Result<WasmSession, JsError> {
        let context = self
            .context
            .take()
            .ok_or_else(|| JsError::new("call prepare() before build()"))?;

        let info = self.compute_model_info()?;
        let ModelCustomInfo::Brumby(custom) = info.custom else {
            return Err(JsError::new("Expected Brumby/PowerCoder model"));
        };
        let use_bias = !custom.has_qk_norm;

        // ── Embed ──
        let embed = Embed {
            w: self
                .embed_cpu
                .take()
                .ok_or_else(|| JsError::new("missing model.embed_tokens.weight"))?,
        };

        // ── Head ──
        let head = Head {
            ln: self.take_rms_norm("model.norm", use_bias, &context)?,
            w: self.take_matrix("lm_head.weight")?,
        };

        // Sync after head
        let submission_index = Some(context.queue.submit(None));
        _ = context.device.poll(wgpu::PollType::Wait {
            submission_index,
            timeout: None,
        });

        // ── Layers ──
        let mut layers = vec![];
        for layer_idx in 0..info.num_layer {
            let l = format!("model.layers.{layer_idx}");

            let input_ln =
                self.take_rms_norm(&format!("{l}.input_layernorm"), use_bias, &context)?;

            let att = Att {
                q_proj: self.take_matrix(&format!("{l}.self_attn.q_proj.weight"))?,
                k_proj: self.take_matrix(&format!("{l}.self_attn.k_proj.weight"))?,
                v_proj: self.take_matrix(&format!("{l}.self_attn.v_proj.weight"))?,
                o_proj: self.take_matrix(&format!("{l}.self_attn.o_proj.weight"))?,
                g_proj: self.take_matrix(&format!("{l}.self_attn.g_proj.weight"))?,
                q_norm: if custom.has_qk_norm {
                    Some(self.take_rms_norm(
                        &format!("{l}.self_attn.q_norm"),
                        false,
                        &context,
                    )?)
                } else {
                    None
                },
                k_norm: if custom.has_qk_norm {
                    Some(self.take_rms_norm(
                        &format!("{l}.self_attn.k_norm"),
                        false,
                        &context,
                    )?)
                } else {
                    None
                },
                q_bias: self.take_optional_vector(
                    &format!("{l}.self_attn.q_proj.bias"),
                    use_bias,
                )?,
                k_bias: self.take_optional_vector(
                    &format!("{l}.self_attn.k_proj.bias"),
                    use_bias,
                )?,
                v_bias: self.take_optional_vector(
                    &format!("{l}.self_attn.v_proj.bias"),
                    use_bias,
                )?,
                o_bias: self.take_optional_vector(
                    &format!("{l}.self_attn.o_proj.bias"),
                    use_bias,
                )?,
                g_bias: self.take_optional_vector(
                    &format!("{l}.self_attn.g_proj.bias"),
                    use_bias,
                )?,
            };

            let post_att_ln = self.take_rms_norm(
                &format!("{l}.post_attention_layernorm"),
                use_bias,
                &context,
            )?;

            let ffn = if custom.gated_ffn {
                Ffn::Gated {
                    gate_proj: self.take_matrix(&format!("{l}.mlp.gate_proj.weight"))?,
                    up_proj: self.take_matrix(&format!("{l}.mlp.up_proj.weight"))?,
                    down_proj: self.take_matrix(&format!("{l}.mlp.down_proj.weight"))?,
                }
            } else {
                Ffn::Dense {
                    c_fc: self.take_matrix(&format!("{l}.mlp.c_fc.weight"))?,
                    c_fc_bias: self
                        .take_optional_vector(&format!("{l}.mlp.c_fc.bias"), use_bias)?,
                    c_proj: self.take_matrix(&format!("{l}.mlp.c_proj.weight"))?,
                    c_proj_bias: self
                        .take_optional_vector(&format!("{l}.mlp.c_proj.bias"), use_bias)?,
                }
            };

            // GPU sync after each layer to release staging buffers
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
            });
        }

        let tensor = ModelTensor {
            embed,
            head,
            layers,
        };
        let model = Model {
            context: context.clone(),
            info: info.clone(),
            rescale: Model::DEFAULT_RESCALE,
            sep: Model::DEFAULT_SEP,
            tensor,
        };

        let num_vocab = info.num_vocab;
        let bundle = BrumbyBundle::new(model, 1);
        let runtime = SimpleRuntime::new(bundle);

        Ok(WasmSession {
            context,
            runtime,
            num_vocab,
            token_chunk_size: self.token_chunk_size.max(32) as usize,
            info,
        })
    }
}

// ── Private helpers on the builder ─────────────────────────

impl WasmSessionBuilder {
    /// Upload a single tensor to GPU (or CPU for embed).
    fn upload_tensor(
        &mut self,
        name: &str,
        reader_tensor: ReaderTensor<'_>,
        context: &Context,
    ) -> Result<(), JsError> {
        let shape = &reader_tensor.1;
        let is_1d = shape.len() <= 1;

        // Embedding weight: pad and keep on CPU.
        if name == "model.embed_tokens.weight" {
            let cpu = TensorCpu::<f16>::from_reader(reader_tensor)
                .map_err(|e| JsError::new(&format!("Tensor error ({name}): {e}")))?
                .pad(PAD_MAT);
            self.embed_cpu = Some(cpu);
            return Ok(());
        }

        // Head weight: pad, upload to GPU as Matrix::Fp16.
        if name == "lm_head.weight" {
            let gpu = TensorCpu::<f16>::from_reader(reader_tensor)
                .map_err(|e| JsError::new(&format!("Tensor error ({name}): {e}")))?
                .pad(PAD_MAT)
                .to(context);
            self.matrices
                .insert(name.to_string(), Matrix::Fp16(gpu));
            return Ok(());
        }

        // Minimum padding to ensure matmul dispatch sizes are non-zero.
        // matmul_vec processes 4 output elements per workgroup, so output dim must be >= 4.
        const PAD_VEC: [usize; 4] = [4, 1, 1, 1];
        const PAD_WEIGHT: [usize; 4] = [4, 4, 1, 1];

        if is_1d {
            // Vector (norm weight / bias): upload as f16, padded to multiple of 4.
            let gpu = TensorCpu::<f16>::from_reader(reader_tensor)
                .map_err(|e| JsError::new(&format!("Tensor error ({name}): {e}")))?
                .pad(PAD_VEC)
                .to(context);
            self.gpu_vectors.insert(name.to_string(), gpu);
        } else {
            // Weight matrix: upload as f16 (padded to multiple of 4), optionally quantize.
            let gpu: TensorGpu<f16, ReadWrite> = TensorCpu::<f16>::from_reader(reader_tensor)
                .map_err(|e| JsError::new(&format!("Tensor error ({name}): {e}")))?
                .pad(PAD_WEIGHT)
                .to(context);

            let layer_idx = Self::extract_layer_index(name);
            let should_quantize = layer_idx
                .map(|idx| idx < self.quant_layers as usize)
                .unwrap_or(false);

            if should_quantize {
                let matrix = Matrix::quant_u8(&gpu)
                    .map_err(|e| JsError::new(&format!("Quant error ({name}): {e}")))?;
                self.matrices.insert(name.to_string(), matrix);
                // gpu (f16 original) is dropped, freeing GPU memory.
            } else {
                self.matrices
                    .insert(name.to_string(), Matrix::Fp16(gpu));
            }
        }

        Ok(())
    }

    /// Extract layer index from tensor names like "model.layers.5.self_attn.q_proj.weight".
    fn extract_layer_index(name: &str) -> Option<usize> {
        let rest = name.strip_prefix("model.layers.")?;
        let dot = rest.find('.')?;
        rest[..dot].parse().ok()
    }

    /// Compute ModelInfo from accumulated tensor metadata.
    fn compute_model_info(&self) -> Result<ModelInfo, JsError> {
        let all_names: Vec<String> = self.weight_map.keys().cloned().collect();
        let meta_reader = MetadataReader {
            all_names,
            shapes: self.tensor_meta.clone(),
        };
        Loader::<MetadataReader>::info(&meta_reader)
            .map_err(|e| JsError::new(&format!("Model info error: {e}")))
    }

    fn take_vector(&mut self, name: &str) -> Result<TensorGpu<f16, ReadWrite>, JsError> {
        self.gpu_vectors
            .remove(name)
            .ok_or_else(|| JsError::new(&format!("missing vector tensor: {name}")))
    }

    fn take_matrix(&mut self, name: &str) -> Result<Matrix, JsError> {
        self.matrices
            .remove(name)
            .ok_or_else(|| JsError::new(&format!("missing matrix tensor: {name}")))
    }

    fn take_rms_norm(
        &mut self,
        prefix: &str,
        has_bias: bool,
        context: &Context,
    ) -> Result<RmsNorm, JsError> {
        let w = self.take_vector(&format!("{prefix}.weight"))?;
        let b = if has_bias {
            self.take_vector(&format!("{prefix}.bias"))?
        } else {
            context.zeros(w.shape())
        };
        Ok(RmsNorm { w, b })
    }

    fn take_optional_vector(
        &mut self,
        name: &str,
        present: bool,
    ) -> Result<Option<TensorGpu<f16, ReadWrite>>, JsError> {
        if present {
            Ok(Some(self.take_vector(name)?))
        } else {
            Ok(None)
        }
    }
}

// ── WasmSession ────────────────────────────────────────────

/// A WASM-exported session that wraps the full inference pipeline.
#[wasm_bindgen]
pub struct WasmSession {
    context: Context,
    runtime: BrumbyRuntime,
    num_vocab: usize,
    token_chunk_size: usize,
    info: ModelInfo,
}

#[wasm_bindgen]
impl WasmSession {
    /// Build from a single (non-sharded) `.safetensors` file.
    pub async fn create(
        model_data: &[u8],
        token_chunk_size: u32,
        quant_layers: u32,
    ) -> Result<WasmSession, JsError> {
        let model = safetensors::SafeTensors::deserialize(model_data)
            .map_err(|e| JsError::new(&format!("SafeTensors parse error: {e}")))?;
        Self::build_from_reader(model, token_chunk_size, quant_layers).await
    }

    /// Shared builder: takes any Reader, creates GPU context and builds the runtime.
    async fn build_from_reader(
        model: impl crate::runtime::loader::Reader,
        token_chunk_size: u32,
        quant_layers: u32,
    ) -> Result<WasmSession, JsError> {
        let info = Loader::info(&model)
            .map_err(|e| JsError::new(&format!("Model info error: {e}")))?;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await
            .map_err(|e| JsError::new(&format!("WebGPU adapter error: {e}")))?;

        let context = ContextBuilder::new(adapter)
            .auto_limits(&info)
            .build()
            .await
            .map_err(|e| JsError::new(&format!("Context error: {e}")))?;

        let mut quant_map: HashMap<usize, Quant> = HashMap::new();
        for layer in 0..(quant_layers as usize).min(info.num_layer) {
            quant_map.insert(layer, Quant::Int8);
        }

        let builder = ModelBuilder::new(&context, model).quant(quant_map);
        let brumby_model = builder
            .build_brumby()
            .await
            .map_err(|e| JsError::new(&format!("Model build error: {e}")))?;

        let num_vocab = info.num_vocab;
        let bundle = BrumbyBundle::new(brumby_model, 1);
        let runtime = SimpleRuntime::new(bundle);

        Ok(WasmSession {
            context,
            runtime,
            num_vocab,
            token_chunk_size: token_chunk_size.max(32) as usize,
            info,
        })
    }

    /// Run inference on the given tokens, returning softmax probabilities
    /// truncated to the real vocabulary size (unpadded).
    pub async fn infer(&self, tokens: &[u32]) -> Result<Vec<f32>, JsError> {
        let logits = self.infer_raw(tokens).await?;

        let logits_tensor = crate::tensor::TensorCpu::from_data(
            [self.num_vocab, 1, 1, 1],
            &logits[..],
        )
        .map_err(|e| JsError::new(&format!("Tensor error: {e}")))?;

        let probs = softmax::softmax_one(&self.context, logits_tensor)
            .await
            .map_err(|e| JsError::new(&format!("Softmax error: {e}")))?;

        Ok(probs.data().to_vec())
    }

    /// Run inference on the given tokens, returning raw logits (no softmax).
    /// The output length equals `num_vocab` (unpadded).
    pub async fn infer_raw(&self, tokens: &[u32]) -> Result<Vec<f32>, JsError> {
        let token_vec: Vec<Token> = tokens.iter().copied().map(Token::Token).collect();
        let batch = RnnInputBatch::new(token_vec, RnnOption::Last);
        let mut input = RnnInput::new(vec![batch], self.token_chunk_size);

        // Loop until we get output (handles multi-chunk prefill).
        let output = loop {
            let (next_input, output) = self
                .runtime
                .infer(input)
                .await
                .map_err(|e| JsError::new(&format!("Inference error: {e}")))?;

            // If the first batch has non-empty output, we're done.
            if !output[0].is_empty() {
                break output;
            }
            input = next_input;
        };

        let all_logits = output[0].data();
        // Truncate to unpadded vocab size.
        Ok(all_logits[..self.num_vocab].to_vec())
    }

    /// Reset the model state to zeros (for starting a new conversation).
    pub fn reset_state(&self) -> Result<(), JsError> {
        let bundle = self.runtime.bundle();
        // Reset RoPE position counter so new conversation starts at position 0.
        bundle.rope_position.store(0, std::sync::atomic::Ordering::Relaxed);
        let state = bundle.state();
        let init = state.init();
        state
            .load(init, 0)
            .map_err(|e| JsError::new(&format!("State reset error: {e}")))?;
        // Reset sum-of-keys accumulators (deg=2 power retention).
        let state = state.as_any().downcast_ref::<crate::runtime::brumby::State>().unwrap();
        state.reset_sum_of_keys();
        Ok(())
    }

    /// Return the unpadded vocabulary size.
    #[wasm_bindgen(getter)]
    pub fn num_vocab(&self) -> usize {
        self.num_vocab
    }

    /// Return the model version string.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        format!("{:?}", self.info.version)
    }

    /// Return the number of layers.
    #[wasm_bindgen(getter)]
    pub fn num_layer(&self) -> usize {
        self.info.num_layer
    }

    /// Return the embedding dimension.
    #[wasm_bindgen(getter)]
    pub fn num_emb(&self) -> usize {
        self.info.num_emb
    }

    /// Return the number of attention heads.
    #[wasm_bindgen(getter)]
    pub fn num_head(&self) -> usize {
        self.info.num_head
    }

    /// Return the first `count` f32 values of the embed vector for `token_id`.
    /// Useful for verifying that the embed tensor data is intact.
    pub fn debug_embed(&self, token_id: u32, count: u32) -> Vec<f32> {
        let bundle = self.runtime.bundle();
        let embed = &bundle.model.tensor.embed.w;
        let num_emb = embed.shape()[0];
        let data = embed.data();
        let start = num_emb * token_id as usize;
        let end = (start + count as usize).min(start + num_emb).min(data.len());
        data[start..end].iter().map(|v| v.to_f32()).collect()
    }

    /// Read back the first `count` f32 values from a GPU norm weight.
    /// layer=-1 means head norm ("model.norm"), otherwise the input_ln of that layer.
    pub async fn debug_norm_weight(&self, layer: i32, count: u32) -> Result<Vec<f32>, JsError> {
        let bundle = self.runtime.bundle();
        let tensor = if layer < 0 {
            &bundle.model.tensor.head.ln.w
        } else {
            let idx = layer as usize;
            if idx >= bundle.model.tensor.layers.len() {
                return Err(JsError::new("layer index out of bounds"));
            }
            &bundle.model.tensor.layers[idx].input_ln.w
        };
        let cpu = tensor.clone().back().await;
        let n = (count as usize).min(cpu.data().len());
        Ok(cpu.data()[..n].iter().map(|v| v.to_f32()).collect())
    }

    /// Return a JSON string with full model info including custom (Brumby/PowerCoder) fields.
    pub fn debug_info(&self) -> String {
        let ModelCustomInfo::Brumby(custom) = self.info.custom else {
            return format!("{{\"error\": \"not Brumby\"}}");
        };
        format!(
            r#"{{"version":"{:?}","num_layer":{},"num_emb":{},"num_hidden":{},"num_vocab":{},"num_head":{},"num_kv_head":{},"head_dim":{},"intermediate_size":{},"has_qk_norm":{},"gated_ffn":{},"hidden_act":"{:?}","rope_theta":{}}}"#,
            self.info.version,
            self.info.num_layer,
            self.info.num_emb,
            self.info.num_hidden,
            self.info.num_vocab,
            self.info.num_head,
            custom.num_kv_head,
            custom.head_dim,
            custom.intermediate_size,
            custom.has_qk_norm,
            custom.gated_ffn,
            custom.hidden_act,
            custom.rope_theta(),
        )
    }

}
