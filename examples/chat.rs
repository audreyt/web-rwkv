//! This example shows how to read-back and load model states to archive
//! session management (e.g., retrying) in a conversational application.

use std::{io::Write, path::PathBuf};

use anyhow::Result;
use clap::{Args, Parser};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Lora, ShardedSafeTensors},
        model::{Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant, State},
        softmax::softmax_one,
        brumby, v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tensor::{TensorCpu, TensorInit, TensorShape},
    tokenizer::{BpeTokenizer, Tokenizer},
};

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

enum AnyTokenizer {
    Rwkv(Tokenizer),
    Bpe(BpeTokenizer),
}

impl AnyTokenizer {
    fn encode(&self, input: &[u8]) -> anyhow::Result<Vec<u32>> {
        Ok(match self {
            Self::Rwkv(t) => t.encode(input)?,
            Self::Bpe(t) => t.encode(input)?,
        })
    }

    fn decode(&self, tokens: &[u32]) -> anyhow::Result<Vec<u8>> {
        Ok(match self {
            Self::Rwkv(t) => t.decode(tokens)?,
            Self::Bpe(t) => t.decode(tokens)?,
        })
    }
}

async fn load_tokenizer(model_path: &std::path::Path, version: ModelVersion) -> Result<AnyTokenizer> {
    match version {
        ModelVersion::Brumby => {
            let dir = if model_path.is_dir() {
                model_path.to_path_buf()
            } else {
                model_path.parent().unwrap().to_path_buf()
            };
            let tok_path = dir.join("tokenizer.json");
            let file = File::open(&tok_path).await?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents).await?;
            Ok(AnyTokenizer::Bpe(BpeTokenizer::new(&contents)?))
        }
        _ => {
            let file = File::open("assets/vocab/rwkv_vocab_v20230424.json").await?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents).await?;
            Ok(AnyTokenizer::Rwkv(Tokenizer::new(&contents)?))
        }
    }
}

async fn load_prompt(path: Option<PathBuf>) -> Result<Prompt> {
    match path {
        Some(path) => {
            let file = File::open(path).await?;
            let mut reader = BufReader::new(file);
            let mut contents = String::new();
            reader.read_to_string(&mut contents).await?;
            Ok(serde_json::from_str(&contents)?)
        }
        None => Ok(Prompt {
            user: String::from("User"),
            bot: String::from("Assistant"),
            intro: String::new(),
            text: vec![
                [
                    String::from("Hi!"),
                    String::from("Hello! I'm your AI assistant. I'm here to help you with various tasks, such as answering questions, brainstorming ideas, drafting emails, writing code, providing advice, and much more.")
                ]
            ],
        }),
    }
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
    #[arg(short, long, action)]
    turbo: bool,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(short, long, action)]
    adapter: bool,
    #[arg(short, long, value_name = "FILE")]
    prompt: Option<PathBuf>,
    #[command(flatten)]
    sampler: Sampler,
}

#[derive(Debug, Serialize, Deserialize)]
struct Prompt {
    user: String,
    bot: String,
    intro: String,
    text: Vec<[String; 2]>,
}

impl Prompt {
    fn build(&self) -> String {
        let user = self.user.trim();
        let bot = self.bot.trim();
        let intro = self.intro.trim();
        let text = self
            .text
            .iter()
            .map(|turn| {
                let user_text = turn[0].trim();
                let bot_text = turn[1].trim();
                format!("{user}: {user_text}\n\n{bot}: {bot_text}\n\n")
            })
            .join("");
        format!("{intro}\n\n{text}")
            .replace("{user}", user)
            .replace("{bot}", bot)
    }
}

#[derive(Debug, Clone, Args)]
struct Sampler {
    #[arg(long, default_value_t = 0.5)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    temp: f32,
}

impl Sampler {
    pub fn sample(&self, probs: &[f32]) -> u32 {
        let sorted: Vec<_> = probs
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u32
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("chat", log::LevelFilter::Info)
        .init()?;
    let cli = Cli::parse();

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
        let p = cli.model.parent().unwrap().join("model.safetensors.index.json");
        (p.exists(), p)
    };

    // Memory-map model data. For sharded models, map each shard file.
    let shard_mmaps: Vec<(String, Mmap)>;
    let single_mmap: Mmap;
    let index_json: String;

    let (context, info, runtime, state): (Context, ModelInfo, Box<dyn Runtime<Rnn>>, Box<dyn State>) = if is_sharded {
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

        let (runtime, state): (Box<dyn Runtime<Rnn>>, Box<dyn State>) = match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                let bundle = brumby::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
        };

        (context, info, runtime, state)
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

        let (runtime, state): (Box<dyn Runtime<Rnn>>, Box<dyn State>) = match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                let bundle = brumby::Bundle::<f16>::new(model, 1);
                let state = bundle.state();
                let runtime = TokioRuntime::new(bundle).await;
                (Box::new(runtime), Box::new(state))
            }
        };

        (context, info, runtime, state)
    };

    let tokenizer = load_tokenizer(&cli.model, info.version).await?;

    println!("\n\nInstructions:\n\n+: Alternative reply\n-: Exit chatting\n\n------------");

    // run initial prompt
    let prompt = load_prompt(cli.prompt).await?;
    let tokens = tokenizer.encode(prompt.build().as_bytes())?;
    let mut inference = RnnInput::new(
        vec![RnnInputBatch::new(tokens, RnnOption::Last)],
        cli.token_chunk_size,
    );

    loop {
        let input = inference.clone();
        let (input, output) = runtime.infer(input).await?;
        inference = input;

        if output[0].size() > 0 {
            assert_eq!(inference.batches[0].tokens.len(), 0);
            break;
        }
    }

    print!("{}", prompt.build());
    std::io::stdout().flush()?;

    // read back initial state
    let mut backed = state.back(0).await?;
    let mut last_user_text = String::from("Hi!");
    let mut last_tokens = vec![];

    // main conversation loop
    loop {
        let mut model_text = String::new();
        let mut user_text = String::new();

        print!("{}: ", prompt.user);
        std::io::stdout().flush()?;

        while user_text.is_empty() {
            std::io::stdin().read_line(&mut user_text)?;
            user_text = user_text.trim().into();
        }

        match user_text.as_str() {
            "-" => break,
            "+" => {
                // retry: reset the prompt and state to the last turn
                user_text.clone_from(&last_user_text);
                inference.batches[0] = RnnInputBatch::new(last_tokens.clone(), RnnOption::Last);
                state.load(backed.clone(), 0)?;
            }
            _ => {
                last_user_text.clone_from(&user_text);
                last_tokens.clone_from(&inference.batches[0].tokens);
                backed = state.back(0).await?;
            }
        }

        print!("\n{}:", prompt.bot);
        std::io::stdout().flush()?;

        let prompt = format!("{}: {}\n\n{}:", prompt.user, user_text, prompt.bot);
        let tokens = tokenizer.encode(prompt.as_bytes())?;
        inference.batches[0].append(tokens);

        // inference loop: read the user prompt and generate until the stop token "\n\n"
        loop {
            let input = inference.clone();
            let (input, output) = runtime.infer(input).await?;
            inference = input;

            let output = output[0].0.clone();
            let shape = output.shape();
            if output.size() == 0 {
                // we are not finishing reading the prompt
                continue;
            }

            let output = output.to_vec();
            assert_eq!(output.len(), info.num_vocab_padded());

            let output = TensorCpu::from_data(shape, output)?;
            let output = softmax_one(&context, output).await?;

            let token = cli.sampler.sample(&output);
            let decoded = tokenizer.decode(&[token])?;
            let word = String::from_utf8_lossy(&decoded);

            model_text += &word;
            print!("{}", word);
            std::io::stdout().flush()?;

            inference.batches[0] = RnnInputBatch::new(vec![token], RnnOption::Last);

            if model_text.contains("\n\n") {
                break;
            }
        }
    }

    Ok(())
}
