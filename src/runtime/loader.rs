use std::{borrow::Cow, collections::HashMap};

use half::f16;
use itertools::Itertools;
use regex::Regex;
use safetensors::{Dtype, SafeTensorError, SafeTensors};
use thiserror::Error;
use web_rwkv_derive::{Deref, DerefMut};

use super::model::{ModelCustomInfo, ModelInfo, ModelVersion, Quant};
use crate::{
    context::Context,
    num::Scalar,
    tensor::{
        kind::ReadWrite,
        matrix::Matrix,
        ops::{Activation, TensorOp},
        shape::{Shape, TensorDimension},
        TensorCpu, TensorError, TensorErrorKind, TensorGpu, TensorInit, TensorInto, TensorReshape,
        TensorShape,
    },
};

pub const PAD_VEC: [usize; 4] = [8, 1, 1, 1];
pub const PAD_MAT: [usize; 4] = [8, 8, 1, 1];

#[derive(Debug, Error)]
pub enum LoaderError {
    #[error("invalid model version")]
    InvalidVersion,
    #[error("tensor error")]
    TensorError(#[from] TensorError),
    #[error("failed to load safe tensor")]
    SafeTensor(#[from] safetensors::SafeTensorError),
    #[error("failed to parse int")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("failed to parse regex")]
    RegexError(#[from] regex::Error),
}

pub type ReaderTensor<'a> = (Dtype, Vec<usize>, Cow<'a, [u8]>);

/// Interface accessing a safetensors data blob.
pub trait Reader {
    fn names(&self) -> Vec<&str>;
    fn contains(&self, name: &str) -> bool;
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError>;
    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError>;
}

impl Reader for SafeTensors<'_> {
    #[inline]
    fn names(&self) -> Vec<&str> {
        self.names().into_iter().map(AsRef::as_ref).collect()
    }

    #[inline]
    fn contains(&self, name: &str) -> bool {
        self.names().contains(&name)
    }

    #[inline]
    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        Ok(self.tensor(name)?.shape().to_vec())
    }

    #[inline]
    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        let tensor = SafeTensors::tensor(self, name)?;
        let shape = tensor.shape().to_vec();
        let data = tensor.data().into();
        Ok((tensor.dtype(), shape, data))
    }
}

/// A reader that spans multiple safetensors shard files.
///
/// Brumby 14B (and other large HuggingFace models) distribute tensors across
/// multiple `.safetensors` files with a `model.safetensors.index.json` mapping
/// tensor names to shard filenames.
pub struct ShardedSafeTensors<'a> {
    /// Map from tensor name to shard index.
    index: HashMap<String, usize>,
    /// Loaded shard SafeTensors instances.
    shards: Vec<SafeTensors<'a>>,
}

impl<'a> ShardedSafeTensors<'a> {
    /// Construct from an index JSON string and pre-loaded shard data.
    ///
    /// `shard_files` maps shard filenames (e.g. `"model-00001-of-00004.safetensors"`)
    /// to their raw bytes. The index JSON's `weight_map` is used to route tensor
    /// lookups to the correct shard.
    pub fn new(
        index_json: &str,
        shard_files: &[(&str, &'a [u8])],
    ) -> Result<Self, LoaderError> {
        #[derive(serde::Deserialize)]
        struct IndexJson {
            weight_map: HashMap<String, String>,
        }

        let parsed: IndexJson =
            serde_json::from_str(index_json).map_err(|_| LoaderError::InvalidVersion)?;

        // Build a map from shard filename to index in the shards vec.
        let mut filename_to_idx: HashMap<String, usize> = HashMap::new();
        let mut shards = Vec::new();
        for &(filename, data) in shard_files {
            let idx = shards.len();
            filename_to_idx.insert(filename.to_string(), idx);
            shards.push(SafeTensors::deserialize(data)?);
        }

        // Build tensor name -> shard index map.
        let mut index = HashMap::new();
        for (tensor_name, shard_filename) in parsed.weight_map {
            let shard_idx = filename_to_idx
                .get(&shard_filename)
                .ok_or(LoaderError::InvalidVersion)?;
            index.insert(tensor_name, *shard_idx);
        }

        Ok(Self { index, shards })
    }
}

impl Reader for ShardedSafeTensors<'_> {
    fn names(&self) -> Vec<&str> {
        self.index.keys().map(|s| s.as_str()).collect()
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>, SafeTensorError> {
        let shard_idx = self
            .index
            .get(name)
            .ok_or(SafeTensorError::TensorNotFound(name.to_string()))?;
        Ok(self.shards[*shard_idx].tensor(name)?.shape().to_vec())
    }

    fn tensor(&self, name: &str) -> Result<ReaderTensor<'_>, SafeTensorError> {
        let shard_idx = self
            .index
            .get(name)
            .ok_or(SafeTensorError::TensorNotFound(name.to_string()))?;
        let tensor = self.shards[*shard_idx].tensor(name)?;
        let shape = tensor.shape().to_vec();
        let data = tensor.data().into();
        Ok((tensor.dtype(), shape, data))
    }
}

pub trait TensorFromReader<T: Scalar> {
    /// Create a tensor from safetensors reader.
    fn from_reader(reader: ReaderTensor) -> Result<TensorCpu<T>, TensorError>;
}

impl<T: Scalar> TensorFromReader<T> for TensorCpu<T> {
    fn from_reader((dt, shape, data): ReaderTensor) -> Result<Self, TensorError> {
        let shape = Shape::from_slice_rev(&shape)?;

        // Fast path: dtype matches exactly.
        if T::DATA_TYPE == dt {
            return match data {
                Cow::Borrowed(data) => Self::from_data(shape, bytemuck::cast_slice(data)),
                Cow::Owned(data) => {
                    let data = bytemuck::cast_slice(&data);
                    let data = Cow::Owned(data.to_vec());
                    Self::from_data(shape, data)
                }
            };
        }

        // Conversion: f32 tensor → f16 target.
        if T::DATA_TYPE == Dtype::F16 && dt == Dtype::F32 {
            let f32_data: &[f32] = bytemuck::cast_slice(&data);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                .collect();
            return Self::from_data(shape, bytemuck::cast_slice(&f16_bytes));
        }

        Err(TensorErrorKind::Type)?
    }
}

/// A LoRA that adds to the model when loading.
#[derive(Clone)]
pub struct Lora<R> {
    /// Binary safetensors LoRA content.
    pub data: R,
    /// A list of LoRA blend patterns.
    /// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
    /// When applying the patterns, they are applied in order.
    pub blend: LoraBlend,
}

/// A list of LoRA blend patterns.
#[derive(Debug, Default, Clone, Deref, DerefMut)]
pub struct LoraBlend(pub Vec<LoraBlendPattern>);

impl LoraBlend {
    /// Build a blend pattern that replaces all vectors, and adds to all matrices with `alpha`.
    #[inline]
    pub fn full(alpha: f32) -> Self {
        Self::default().add_nominal(1.0).add_matrices(alpha)
    }

    /// Add a blend pattern that interpolates tensors with factor `alpha` from 0 to 1.
    #[inline]
    pub fn add_nominal(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(r".+", alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices with `alpha`.
    #[inline]
    pub fn add_matrices(mut self, alpha: f32) -> Self {
        let pattern = LoraBlendPattern::new(
            r"blocks\.([0-9]+)\.(att|ffn)\.(key|value|receptance|gate|output)\.weight",
            alpha,
        )
        .unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that interpolates tensors in a layer with factor `alpha` from 0 to 1.
    pub fn add_layer_nominal(mut self, layer: usize, alpha: f32) -> Self {
        let pattern = format!(r"blocks\.{layer}");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }

    /// Add a blend pattern that adds to all matrices in a layer with `alpha`.
    pub fn add_layer_matrices(mut self, layer: usize, alpha: f32) -> Self {
        let pattern =
            format!(r"blocks\.{layer}\.(att|ffn)\.(key|value|receptance|gate|output)\.weight");
        let pattern = LoraBlendPattern::new(&pattern, alpha).unwrap();
        self.push(pattern);
        self
    }
}

/// A blend pattern is a regex that matches the name of multiple tensors, and a blend factor.
#[derive(Debug, Clone)]
pub struct LoraBlendPattern {
    /// A regex pattern that matches tensors in the model.
    pattern: Regex,
    /// The blend factor.
    alpha: f32,
}

impl LoraBlendPattern {
    #[inline]
    pub fn new(pattern: &str, alpha: f32) -> Result<Self, LoaderError> {
        Ok(Self {
            pattern: Regex::new(pattern)?,
            alpha,
        })
    }

    #[inline]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

struct LoraVector {
    tensor: TensorGpu<f16, ReadWrite>,
    alpha: f32,
}

struct LoraMatrix {
    x: TensorGpu<f16, ReadWrite>,
    y: TensorGpu<f16, ReadWrite>,
    rank: usize,
    alpha: f32,
}

#[derive(Clone)]
pub struct Loader<R> {
    pub context: Context,
    pub model: R,
    pub lora: Vec<Lora<R>>,
}

impl<R: Reader> Loader<R> {
    pub fn info(model: &R) -> Result<ModelInfo, LoaderError> {
        // Check for PowerCoder (power retention + StarCoder2 FFN, no QK norms).
        let powercoder = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.g_proj.weight",
            "model.layers.0.mlp.c_fc.weight",
            "model.layers.0.mlp.c_proj.weight",
        ]
        .iter()
        .all(|n| model.contains(n))
            && !model.contains("model.layers.0.self_attn.q_norm.weight");

        if powercoder {
            return Self::info_powercoder(model);
        }

        // Check for Brumby (HuggingFace-style tensor naming with power retention).
        let brumby = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.self_attn.g_proj.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.self_attn.k_norm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
        .into_iter()
        .all(|name| model.contains(name));

        if brumby {
            return Self::info_brumby(model);
        }

        let num_layer = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "blocks.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };

        let embed = model.shape("emb.weight")?;
        let ffn = model.shape("blocks.0.ffn.key.weight")?;

        let v4 = [
            "blocks.0.att.time_decay",
            "blocks.0.att.time_first",
            "blocks.0.att.time_mix_k",
            "blocks.0.att.time_mix_v",
            "blocks.0.att.time_mix_r",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v5 = [
            "blocks.0.att.gate.weight",
            "blocks.0.att.ln_x.weight",
            "blocks.0.att.ln_x.bias",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v6 = [
            "blocks.0.att.time_mix_x",
            "blocks.0.att.time_mix_w",
            "blocks.0.att.time_mix_k",
            "blocks.0.att.time_mix_v",
            "blocks.0.att.time_mix_r",
            "blocks.0.att.time_mix_g",
            "blocks.0.att.time_mix_w1",
            "blocks.0.att.time_mix_w2",
            "blocks.0.att.time_decay_w1",
            "blocks.0.att.time_decay_w2",
            "blocks.0.ffn.time_mix_k",
            "blocks.0.ffn.time_mix_r",
        ]
        .into_iter()
        .all(|name| model.contains(name));
        let v7 = [
            "blocks.0.att.x_r",
            "blocks.0.att.x_w",
            "blocks.0.att.x_k",
            "blocks.0.att.x_v",
            "blocks.0.att.x_a",
            "blocks.0.att.x_g",
            "blocks.0.att.w0",
            "blocks.0.att.w1",
            "blocks.0.att.w2",
            "blocks.0.att.a0",
            "blocks.0.att.a1",
            "blocks.0.att.a2",
            "blocks.0.att.g1",
            "blocks.0.att.g2",
            "blocks.0.att.r_k",
            "blocks.0.att.k_k",
            "blocks.0.att.k_a",
        ]
        .into_iter()
        .all(|name| model.contains(name));

        let version = match (v4, v5, v6, v7) {
            (true, false, false, false) => ModelVersion::V4,
            (_, true, false, false) => ModelVersion::V5,
            (_, _, true, false) => ModelVersion::V6,
            (_, _, _, true) => ModelVersion::V7,
            _ => return Err(LoaderError::InvalidVersion),
        };

        let num_emb = embed[1];
        let num_hidden = ffn[0];
        let num_vocab = embed[0];

        let num_head = match version {
            ModelVersion::V4 => 1,
            ModelVersion::V5 | ModelVersion::V6 => model.shape("blocks.0.att.time_first")?[0],
            ModelVersion::V7 => model.shape("blocks.0.att.r_k")?[0],
            ModelVersion::Brumby => unreachable!(),
        };

        let custom = match version {
            ModelVersion::V6 => {
                let time_mix = model.shape("blocks.0.att.time_mix_w1")?[0] / 5;
                let time_decay = model.shape("blocks.0.att.time_decay_w1")?[0];
                ModelCustomInfo::V6(super::v6::CustomInfo {
                    time_mix,
                    time_decay,
                })
            }
            ModelVersion::V7 => {
                let w = model.shape("blocks.0.att.w1")?[0];
                let a = model.shape("blocks.0.att.a1")?[0];
                let g = model.shape("blocks.0.att.g1")?[0];
                let v = model.shape("blocks.1.att.v1")?[0];
                ModelCustomInfo::V7(super::v7::CustomInfo { w, a, g, v })
            }
            _ => ModelCustomInfo::None,
        };

        Ok(ModelInfo {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
            custom,
        })
    }

    /// Extract model info for Brumby (HuggingFace-style tensor naming).
    fn info_brumby(model: &R) -> Result<ModelInfo, LoaderError> {
        let num_layer = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "model.layers.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };

        // embed_tokens.weight shape: [vocab_size, hidden_size]
        let embed = model.shape("model.embed_tokens.weight")?;
        let num_vocab = embed[0];
        let num_emb = embed[1];

        // gate_proj.weight shape: [intermediate_size, hidden_size]
        let ffn = model.shape("model.layers.0.mlp.gate_proj.weight")?;
        let intermediate_size = ffn[0];

        // q_proj.weight shape: [num_heads * head_dim, hidden_size]
        let q_shape = model.shape("model.layers.0.self_attn.q_proj.weight")?;
        // k_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
        let k_shape = model.shape("model.layers.0.self_attn.k_proj.weight")?;

        // q_norm.weight shape: [head_dim]
        let q_norm_shape = model.shape("model.layers.0.self_attn.q_norm.weight")?;
        let head_dim = q_norm_shape[0];

        let num_head = q_shape[0] / head_dim;
        let num_kv_head = k_shape[0] / head_dim;

        let custom = ModelCustomInfo::Brumby(
            super::brumby::CustomInfo {
                num_kv_head,
                head_dim,
                intermediate_size,
                has_qk_norm: true,
                gated_ffn: true,
                hidden_act: Activation::Silu,
                rope_theta_bits: 0,
                power_deg: 1,
            }
            .with_rope_theta(1_000_000.0),
        );

        Ok(ModelInfo {
            version: ModelVersion::Brumby,
            num_layer,
            num_emb,
            num_hidden: intermediate_size,
            num_vocab,
            num_head,
            custom,
        })
    }

    fn info_powercoder(model: &R) -> Result<ModelInfo, LoaderError> {
        let num_layer = {
            let mut r: usize = 0;
            for i in model.names() {
                const PREFIX: &str = "model.layers.";
                if let Some(i) = i.strip_prefix(PREFIX) {
                    let i = &i[..i.find('.').unwrap_or(0)];
                    r = r.max(i.parse::<usize>()?)
                }
            }
            r + 1
        };

        // embed_tokens.weight shape: [vocab_size, hidden_size]
        let embed = model.shape("model.embed_tokens.weight")?;
        let num_vocab = embed[0];
        let num_emb = embed[1];

        // c_fc.weight shape: [intermediate_size, hidden_size]
        let ffn = model.shape("model.layers.0.mlp.c_fc.weight")?;
        let intermediate_size = ffn[0];

        // q_proj.weight shape: [num_heads * head_dim, hidden_size]
        let q_shape = model.shape("model.layers.0.self_attn.q_proj.weight")?;
        // k_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
        let k_shape = model.shape("model.layers.0.self_attn.k_proj.weight")?;

        // g_proj.weight shape: [num_kv_heads, hidden_size] (one gate per KV head group)
        let g_shape = model.shape("model.layers.0.self_attn.g_proj.weight")?;
        let num_kv_head = g_shape[0];
        let head_dim = k_shape[0] / num_kv_head;
        let num_head = q_shape[0] / head_dim;

        let custom = ModelCustomInfo::Brumby(
            super::brumby::CustomInfo {
                num_kv_head,
                head_dim,
                intermediate_size,
                has_qk_norm: false,
                gated_ffn: false,
                hidden_act: Activation::Gelu,
                rope_theta_bits: 0,
                power_deg: 2,
            }
            .with_rope_theta(10_000.0),
        );

        Ok(ModelInfo {
            version: ModelVersion::Brumby,
            num_layer,
            num_emb,
            num_hidden: intermediate_size,
            num_vocab,
            num_head,
            custom,
        })
    }

    /// Load all lora and blend factors about the vector with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    fn lora_vectors(&self, name: impl AsRef<str>) -> Result<Vec<LoraVector>, LoaderError> {
        let context = &self.context;
        let name = name.as_ref();

        let mut vectors = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .rfind(|blend| blend.pattern.is_match(name))
            else {
                continue;
            };

            let Ok(tensor) = lora.data.tensor(name) else {
                continue;
            };
            let tensor = TensorCpu::from_reader(tensor)?.to(context);
            let alpha = blend.alpha;
            vectors.push(LoraVector { tensor, alpha });

            log::info!("vector (LoRA) {name}, alpha: {alpha}");
        }
        Ok(vectors)
    }

    /// Load all lora and blend factors about the matrix with a given name.
    /// In each LoRA, only the last matched pattern is loaded.
    fn lora_matrices(&self, name: impl AsRef<str>) -> Result<Vec<LoraMatrix>, LoaderError> {
        let context = &self.context;
        let name = name.as_ref();

        let mut matrices = vec![];
        for lora in self.lora.iter() {
            let Some(blend) = lora
                .blend
                .iter()
                .rfind(|blend| blend.pattern.is_match(name))
            else {
                continue;
            };

            let name = name.split('.').filter(|x| !x.contains("weight")).join(".");
            let Ok(x) = lora.data.tensor(&format!("{name}.lora.0")) else {
                continue;
            };
            let Ok(y) = lora.data.tensor(&format!("{name}.lora.1")) else {
                continue;
            };

            let rank = x.1[1];
            let alpha = blend.alpha;
            let x = TensorCpu::from_reader(x)?.to(context);
            let y = TensorCpu::from_reader(y)?.to(context);
            matrices.push(LoraMatrix { x, y, rank, alpha });

            log::info!("matrix (LoRA) {name}, alpha: {alpha}, rank: {rank}");
        }
        Ok(matrices)
    }

    pub fn tensor_shape(&self, name: impl AsRef<str>) -> Result<Shape, LoaderError> {
        let shape = self.model.shape(name.as_ref())?;
        Ok(Shape::from_slice_rev(&shape)?)
    }

    pub fn load_vector_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        let op = TensorOp::activate(&tensor, Activation::OppositeExp)?;
        ops.push(op);

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_exp_exp_f32(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f32, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            // .map(|x| -x.to_f32().exp())
            // .map(|x| x.exp())
            .map(|x| x.to_f32())
            .reshape(
                TensorDimension::Auto,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_vectors(name)? {
            let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

            let shape = lora.tensor.shape();
            let tensor = tensor.reshape(
                TensorDimension::Size(shape[0]),
                TensorDimension::Size(shape[1]),
                TensorDimension::Size(shape[2]),
                TensorDimension::Size(shape[3]),
            )?;

            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        let op = TensorOp::activate(&tensor, Activation::StableExp)?;
        ops.push(op);

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_vector_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let lora = self.lora_vectors(name.as_ref())?;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = if lora.is_empty() {
            TensorCpu::from_reader(tensor)?
                .reshape(
                    TensorDimension::Auto,
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                )?
                .to(context)
        } else {
            let tensor_f32: TensorGpu<f32, _> = TensorCpu::<f16>::from_reader(tensor)?
                .map(|x| x.to_f32())
                .reshape(
                    TensorDimension::Auto,
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                    TensorDimension::Size(1),
                )?
                .to(context);
            let tensor_f16: TensorGpu<f16, _> = context.tensor_init(tensor_f32.shape());

            let mut ops = vec![];
            for lora in lora {
                let factor = vec![lora.alpha, 1.0 - lora.alpha, 0.0, 0.0];
                let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;

                let shape = lora.tensor.shape();
                let tensor = tensor_f32.reshape(
                    TensorDimension::Size(shape[0]),
                    TensorDimension::Size(shape[1]),
                    TensorDimension::Size(shape[2]),
                    TensorDimension::Size(shape[3]),
                )?;

                let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
                ops.push(op);
            }

            let op = TensorOp::blit(&tensor_f32, &tensor_f16)?;
            ops.push(op);

            context.queue.submit(context.encode(&TensorOp::List(ops)));
            tensor_f16
        };
        Ok(tensor)
    }

    pub fn load_matrix_f16(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = TensorCpu::from_reader(tensor)?.to(context);

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, &tensor)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_matrix_f16_discount(
        &self,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor: TensorGpu<_, _> = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .to(context);

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, &tensor)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, &tensor)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(tensor)
    }

    pub fn load_in_place_matrix_f16(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
    ) -> Result<(), LoaderError> {
        let context = &self.context;
        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = TensorCpu::from_reader(tensor)?;
        matrix.load(&tensor)?;

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, matrix)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(())
    }

    pub fn load_in_place_matrix_f16_discount(
        &self,
        matrix: &TensorGpu<f16, ReadWrite>,
        name: impl AsRef<str>,
        discount: f32,
    ) -> Result<(), LoaderError> {
        let context = &self.context;

        let tensor = self.model.tensor(name.as_ref())?;
        let tensor = TensorCpu::<f16>::from_reader(tensor)?
            .map(|x| f16::from_f32(discount * x.to_f32()))
            .reshape(
                TensorDimension::Full,
                TensorDimension::Full,
                TensorDimension::Size(1),
                TensorDimension::Size(1),
            )?;
        matrix.load(&tensor)?;

        let mut ops = vec![];
        for lora in self.lora_matrices(name.as_ref())? {
            let factor = vec![discount * lora.alpha / lora.rank as f32, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend_lora(&factor, &lora.x, &lora.y, matrix)?;
            ops.push(op);
        }
        for lora in self.lora_vectors(name.as_ref())? {
            let factor = vec![discount * lora.alpha, 1.0, 0.0, 0.0];
            let factor = context.tensor_from_data([4, 1, 1, 1], factor)?;
            let op = TensorOp::blend(&factor, &lora.tensor, matrix)?;
            ops.push(op);
        }

        context.queue.submit(context.encode(&TensorOp::List(ops)));
        Ok(())
    }

    pub fn load_matrix_f16_padded_cpu(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorCpu<f16>, LoaderError> {
        let (dt, shape, tensor) = self.model.tensor(name.as_ref())?;
        let tensor = TensorCpu::from_reader((dt, shape, tensor))?.pad(PAD_MAT);
        Ok(tensor)
    }

    pub fn load_matrix_f16_padded(
        &self,
        name: impl AsRef<str>,
    ) -> Result<TensorGpu<f16, ReadWrite>, LoaderError> {
        let context = &self.context;
        let (dt, shape, tensor) = self.model.tensor(name.as_ref())?;
        let tensor = TensorCpu::from_reader((dt, shape, tensor))?
            .pad(PAD_MAT)
            .to(context);
        Ok(tensor)
    }

    pub fn load_matrix(&self, name: String, quant: Quant) -> Result<Matrix, LoaderError> {
        let context = &self.context;
        match quant {
            Quant::None => Ok(Matrix::Fp16(self.load_matrix_f16(name)?)),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
            Quant::SF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16(&buffer, &name)?;
                Ok(Matrix::quant_sf4(&buffer, 5.0)?)
            }
        }
    }

    pub fn load_matrix_discount(
        &self,
        name: String,
        quant: Quant,
        discount: f32,
    ) -> Result<Matrix, LoaderError> {
        let context = &self.context;
        match quant {
            Quant::None => Ok(Matrix::Fp16(self.load_matrix_f16_discount(name, discount)?)),
            Quant::Int8 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_u8(&buffer)?)
            }
            Quant::NF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_nf4(&buffer)?)
            }
            Quant::SF4 => {
                let shape = self.tensor_shape(&name)?;
                let buffer = context.tensor_init(shape);
                self.load_in_place_matrix_f16_discount(&buffer, &name, discount)?;
                Ok(Matrix::quant_sf4(&buffer, 5.0)?)
            }
        }
    }
}
