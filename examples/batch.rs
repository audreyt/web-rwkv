//! This example shows that we can run multiple inferences of different length at the same time.

use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use clap::Parser;
#[cfg(not(debug_assertions))]
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
#[cfg(not(debug_assertions))]
use dialoguer::{theme::ColorfulTheme, Select};
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
#[cfg(not(debug_assertions))]
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Layout},
    style::{Color, Modifier, Style, Stylize},
    text::{Span, Text},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use safetensors::SafeTensors;
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::{Loader, Lora, ShardedSafeTensors},
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        softmax::softmax,
        brumby, v4, v5, v6, v7, Runtime, TokioRuntime,
    },
    tokenizer::{BpeTokenizer, Tokenizer},
};

fn sample(probs: &[f32], _top_p: f32) -> u32 {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.total_cmp(y))
        .unwrap()
        .0 as u32
}

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

#[cfg(not(debug_assertions))]
fn setup_terminal() -> Result<Terminal<CrosstermBackend<std::io::Stdout>>> {
    let mut stdout = std::io::stdout();
    enable_raw_mode()?;
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    Ok(Terminal::new(CrosstermBackend::new(stdout))?)
}

#[cfg(not(debug_assertions))]
fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen,)?;
    Ok(terminal.show_cursor()?)
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
    #[arg(short, long, default_value_t = 4)]
    batch: usize,
    #[arg(short, long, action)]
    adapter: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("batch", log::LevelFilter::Info)
        .init()?;
    let cli = Cli::parse();
    let batch = cli.batch;

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

    let (context, info, runtime): (Context, ModelInfo, Box<dyn Runtime<Rnn>>) = if is_sharded {
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
                Box::new(TokioRuntime::new(v4::Bundle::<f16>::new(model, batch)).await)
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                Box::new(TokioRuntime::new(v5::Bundle::<f16>::new(model, batch)).await)
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                Box::new(TokioRuntime::new(v6::Bundle::<f16>::new(model, batch)).await)
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                Box::new(TokioRuntime::new(v7::Bundle::<f16>::new(model, batch)).await)
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                Box::new(TokioRuntime::new(brumby::Bundle::<f16>::new(model, batch)).await)
            }
        };

        (context, info, runtime)
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
                let bundle = v4::Bundle::<f16>::new(model, batch);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, batch);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, batch);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, batch);
                Box::new(TokioRuntime::new(bundle).await)
            }
            ModelVersion::Brumby => {
                let model = builder.build_brumby().await?;
                let bundle = brumby::Bundle::<f16>::new(model, batch);
                Box::new(TokioRuntime::new(bundle).await)
            }
        };

        (context, info, runtime)
    };

    let tokenizer = load_tokenizer(&cli.model, info.version).await?;

    #[cfg(not(debug_assertions))]
    let mut terminal = setup_terminal()?;

    let prompts = [
        "The Eiffel Tower is located in the city of",
        "The name of the capital of Italy is",
        "The Space Needle is located in downtown",
        "人们发现",
    ];
    let mut prompts = prompts.to_vec().repeat(batch.div_ceil(prompts.len()))[..batch]
        .iter()
        .map(|str| String::from_str(str).unwrap())
        .collect_vec();
    let tokens = prompts
        .clone()
        .iter()
        .map(|prompt| tokenizer.encode(prompt.as_bytes()).unwrap())
        .collect_vec();

    let mut inference = RnnInput::new(
        tokens
            .into_iter()
            .map(|tokens| RnnInputBatch::new(tokens, RnnOption::Last))
            .collect(),
        cli.token_chunk_size,
    );

    let mut num_token =
        [100usize, 400, 200, 300].to_vec().repeat(batch.div_ceil(4))[..batch].to_vec();

    loop {
        #[cfg(not(debug_assertions))]
        terminal.draw(|frame| {
            let size = frame.area();

            let block = Block::default().black();
            frame.render_widget(block, size);

            let constraints = (0..batch)
                .map(|_| Constraint::Percentage(100 / batch as u16))
                .collect_vec();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(constraints)
                .split(size);

            let create_block = |title| {
                Block::default()
                    .borders(Borders::ALL)
                    .style(Style::default().fg(Color::Gray))
                    .title(Span::styled(
                        title,
                        Style::default().add_modifier(Modifier::BOLD),
                    ))
            };

            for (index, (text, chunk)) in prompts.iter().zip(chunks.iter()).enumerate() {
                let text = Text::from(text.as_str());
                let text_height_estimation: usize = text
                    .lines
                    .iter()
                    .map(|line| (line.width() / 1.max(chunk.width as usize - 2)).max(1))
                    .sum();
                let scroll =
                    (text_height_estimation as isize - chunk.height as isize + 2).max(0) as u16;
                let paragraph = Paragraph::new(text)
                    .style(Style::default().fg(Color::Gray))
                    .block(create_block(format!("Batch {index}")))
                    .wrap(Wrap { trim: true })
                    .scroll((scroll, 0));
                frame.render_widget(paragraph, *chunk);
            }
        })?;

        #[cfg(debug_assertions)]
        for (index, prompt) in prompts.iter().enumerate() {
            println!("{index}: {prompt}");
        }

        let input = inference.clone();
        let (input, output) = runtime.infer(input).await?;
        inference = input;

        let output = output.iter().map(|batch| batch.0.clone()).collect_vec();
        let output = softmax(&context, output).await?;
        for (index, batch) in output.iter().enumerate() {
            if batch.size() == 0 {
                continue;
            }
            if num_token[index] > 0 {
                let batch = batch.clone().to_vec();
                let token = sample(&batch, 0.5);
                let decoded = tokenizer.decode(&[token])?;
                let word = String::from_utf8_lossy(&decoded);
                inference.batches[index].replace(vec![token]);
                prompts[index].push_str(&word);
                num_token[index] -= 1;
            }
        }

        if num_token.iter().all(|x| *x == 0) {
            break;
        }
    }

    #[cfg(not(debug_assertions))]
    restore_terminal(&mut terminal)?;

    Ok(())
}
