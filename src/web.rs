use std::collections::HashMap;

use half::f16;
use wasm_bindgen::prelude::*;

use crate::{
    context::{Context, ContextBuilder, InstanceExt as _},
    runtime::{
        brumby,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption, Token},
        loader::{Loader, ShardedSafeTensors},
        model::{Bundle as _, ContextAutoLimits as _, ModelBuilder, Quant, State as _},
        softmax, SimpleRuntime,
    },
    tensor::TensorInit as _,
};

type BrumbyBundle = brumby::Bundle<f16>;
type BrumbyRuntime = SimpleRuntime<BrumbyBundle, Rnn, brumby::RnnJob>;

/// Incremental builder for sharded models.
///
/// Holds shard bytes in WASM linear memory so JS can release each
/// `Uint8Array` after calling `add_shard`, keeping peak JS heap small.
#[wasm_bindgen]
pub struct WasmSessionBuilder {
    index_json: String,
    shard_names: Vec<String>,
    shard_data: Vec<Vec<u8>>,
    token_chunk_size: u32,
    quant_layers: u32,
}

#[wasm_bindgen]
impl WasmSessionBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        index_json: &str,
        token_chunk_size: u32,
        quant_layers: u32,
    ) -> WasmSessionBuilder {
        WasmSessionBuilder {
            index_json: index_json.to_string(),
            shard_names: Vec::new(),
            shard_data: Vec::new(),
            token_chunk_size,
            quant_layers,
        }
    }

    /// Copy one shard's bytes into WASM memory. JS can free the buffer afterwards.
    pub fn add_shard(&mut self, name: &str, data: &[u8]) {
        self.shard_names.push(name.to_string());
        self.shard_data.push(data.to_vec());
    }

    /// Consume the builder and produce a ready-to-use `WasmSession`.
    pub async fn build(self) -> Result<WasmSession, JsError> {
        let shard_files: Vec<(&str, &[u8])> = self
            .shard_names
            .iter()
            .zip(self.shard_data.iter())
            .map(|(n, d)| (n.as_str(), d.as_slice()))
            .collect();

        let model = ShardedSafeTensors::new(&self.index_json, &shard_files)
            .map_err(|e| JsError::new(&format!("Sharded model parse error: {e}")))?;

        WasmSession::build_from_reader(model, self.token_chunk_size, self.quant_layers).await
    }
}

/// A WASM-exported session that wraps the full inference pipeline.
#[wasm_bindgen]
pub struct WasmSession {
    context: Context,
    runtime: BrumbyRuntime,
    num_vocab: usize,
    token_chunk_size: usize,
    info: crate::runtime::model::ModelInfo,
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
        let state = bundle.state();
        let init = state.init();
        state
            .load(init, 0)
            .map_err(|e| JsError::new(&format!("State reset error: {e}")))?;
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
}
