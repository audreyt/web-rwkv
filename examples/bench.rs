//! A simple example to benchmark the inference performance of the model.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use instant::{Duration, Instant};
#[cfg(not(debug_assertions))]
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;

use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
#[cfg(feature = "trace")]
use tracing_subscriber::layer::SubscriberExt;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Lora, ShardedSafeTensors},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax_one,
        brumby, v4, v5, v6, v7, Runtime, TokioRuntime,
    },
};

const REPEAT: usize = 5;

async fn create_context(info: &ModelInfo, _auto: bool) -> Result<Context> {
    let instance = wgpu::Instance::default();
    #[cfg(not(debug_assertions))]
    let adapter = if _auto {
        instance
            .adapter(wgpu::PowerPreference::HighPerformance)
            .await?
    } else {
        let backends = wgpu::Backends::all();
        let adapters = instance.enumerate_adapters(backends).await;
        let names = adapters
            .iter()
            .map(|adapter| adapter.get_info())
            .map(|info| format!("{} ({:?})", info.name, info.backend))
            .collect_vec();
        let selection = Select::with_theme(&ColorfulTheme::default())
            .with_prompt("Please select an adapter")
            .default(0)
            .items(&names)
            .interact()?;
        adapters.into_iter().nth(selection).unwrap()
    };
    #[cfg(debug_assertions)]
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

fn sample(probs: &[f32], _top_p: f32) -> u16 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u16
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(short, long, value_name = "FILE")]
    lora: Option<PathBuf>,
    #[arg(short, long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_sf4: usize,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(long, default_value_t = 512)]
    prefill_length: usize,
    #[arg(long, default_value_t = 128)]
    generation_length: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("bench", log::LevelFilter::Info)
        .init()?;
    #[cfg(feature = "trace")]
    {
        let registry = tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default());
        tracing::subscriber::set_global_default(registry)?;
    }

    let cli = Cli::parse();
    let model_path = cli.model.clone();

    // Detect sharded vs single-file model.
    // Sharded models are detected by:
    // 1. --model points to a directory containing model.safetensors.index.json
    // 2. --model points to a .safetensors.index.json file
    let (is_sharded, index_json_path) = if cli.model.is_dir() {
        let p = cli.model.join("model.safetensors.index.json");
        (p.exists(), p)
    } else if cli.model.extension().is_some_and(|e| e == "json") {
        (cli.model.exists(), cli.model.clone())
    } else {
        let p = cli
            .model
            .parent()
            .unwrap()
            .join("model.safetensors.index.json");
        (p.exists(), p)
    };

    // Memory-map model data. For sharded models, map each shard file.
    let shard_mmaps: Vec<(String, Mmap)>;
    let single_mmap: Mmap;
    let index_json: String;

    let (info, context, runtime): (ModelInfo, Context, Box<dyn Runtime<Rnn>>) = if is_sharded {
        let base_dir = index_json_path.parent().unwrap().to_path_buf();

        let mut idx_file = File::open(&index_json_path).await?;
        let mut idx_contents = String::new();
        idx_file.read_to_string(&mut idx_contents).await?;
        index_json = idx_contents;

        // Parse index to find unique shard filenames.
        let parsed: serde_json::Value = serde_json::from_str(&index_json)?;
        let weight_map = parsed["weight_map"].as_object().unwrap();
        let mut shard_filenames: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_filenames.sort();
        shard_filenames.dedup();

        // Memory-map all shard files.
        shard_mmaps = {
            let mut mmaps = Vec::new();
            for filename in &shard_filenames {
                let path = base_dir.join(filename);
                let file = File::open(&path).await?;
                let mmap = unsafe { Mmap::map(&file)? };
                mmaps.push((filename.clone(), mmap));
            }
            mmaps
        };

        let shard_refs: Vec<(&str, &[u8])> = shard_mmaps
            .iter()
            .map(|(name, mmap)| (name.as_str(), mmap.as_ref()))
            .collect();
        let model = ShardedSafeTensors::new(&index_json, &shard_refs)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let context = create_context(&info, cli.adapter).await?;
        log::info!("{:#?}", context.adapter.get_info());

        let quant = (0..cli.quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..cli.quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();

        let builder = ModelBuilder::new(&context, model).quant(quant);

        let runtime: Box<dyn Runtime<Rnn>> = match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                Box::new(TokioRuntime::new(v4::Bundle::<f16>::new(model, 1)).await)
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                Box::new(TokioRuntime::new(v5::Bundle::<f16>::new(model, 1)).await)
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                Box::new(TokioRuntime::new(v6::Bundle::<f16>::new(model, 1)).await)
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                Box::new(TokioRuntime::new(v7::Bundle::<f16>::new(model, 1)).await)
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                Box::new(TokioRuntime::new(brumby::Bundle::<f16>::new(model, 1)).await)
            }
        };

        (info, context, runtime)
    } else {
        let model_path = if cli.model.is_dir() {
            cli.model.join("model.safetensors")
        } else {
            cli.model.clone()
        };
        let file = File::open(&model_path).await?;
        single_mmap = unsafe { Mmap::map(&file)? };

        let model = SafeTensors::deserialize(&single_mmap)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let context = create_context(&info, cli.adapter).await?;
        log::info!("{:#?}", context.adapter.get_info());

        let quant = (0..cli.quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..cli.quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..cli.quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();
        let lora = match cli.lora {
            Some(path) => {
                let file = File::open(path).await?;
                let mut reader = BufReader::new(file);
                let mut data = vec![];
                reader.read_to_end(&mut data).await?;
                Some(data)
            }
            None => None,
        };

        let builder = ModelBuilder::new(&context, model).quant(quant);
        let builder = match &lora {
            Some(data) => {
                let data = SafeTensors::deserialize(data)?;
                let blend = Default::default();
                let lora = Lora { data, blend };
                builder.lora(lora)
            }
            None => builder,
        };

        let runtime: Box<dyn Runtime<Rnn>> = match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                let bundle = brumby::Bundle::<f16>::new(model, 1);
                Box::new(TokioRuntime::new(bundle).await)
            }
        };

        (info, context, runtime)
    };

    fastrand::seed(42);
    let prompt_len = cli.prefill_length;

    println!("| model                                                    | quant_int8 | quant_float4 |    test |            t/s |");
    println!("|----------------------------------------------------------|-----------:|-------------:|--------:|---------------:|");

    let mut prefill = Duration::ZERO;
    for _ in 0..REPEAT {
        let tokens: Vec<u16> = (0..prompt_len)
            .map(|_| fastrand::u16(0..((info.num_vocab - 1) as u16)))
            .collect();
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);
        let instant = Instant::now();
        for _ in 0..prompt_len {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let output = output.to_vec();
                let _ = sample(&output, 0.0);

                let duration = instant.elapsed();
                prefill += duration;
                break;
            }
        }
    }
    let tps = prompt_len as f64 / prefill.as_secs_f64() * REPEAT as f64;
    println!(
        "| {:<56} | {:>10} | {:>12} | {:>7} | {:>14} |",
        model_path.file_name().unwrap().to_string_lossy(),
        cli.quant,
        cli.quant_nf4.max(cli.quant_sf4),
        format!("pp{}", cli.prefill_length.to_string().clone()),
        format!("{:.2}", tps)
    );

    let num_token = cli.generation_length;
    let mut generation = Duration::ZERO;
    for _ in 0..REPEAT {
        let tokens: Vec<u16> = vec![0];
        let prompt = RnnInputBatch::new(tokens, RnnOption::Last);
        let mut prompt = RnnInput::new(vec![prompt], cli.token_chunk_size);
        let instant = Instant::now();
        for _ in 0..num_token {
            let input = prompt.clone();
            let (input, output) = runtime.infer(input).await?;
            prompt = input;

            let output = output[0].0.clone();
            if output.size() > 0 {
                let output = softmax_one(&context, output).await?;
                let output = output.to_vec();
                let token = sample(&output, 0.0);
                prompt.batches[0].push(token);
            }
        }
        generation += instant.elapsed();
    }
    let tps = num_token as f64 / generation.as_secs_f64() * REPEAT as f64;
    println!(
        "| {:<56} | {:>10} | {:>12} | {:>7} | {:>14} |",
        model_path.file_name().unwrap().to_string_lossy(),
        cli.quant,
        cli.quant_nf4.max(cli.quant_sf4),
        format!("tg{}", cli.generation_length.to_string().clone()),
        format!("{:.2}", tps)
    );
    Ok(())
}
