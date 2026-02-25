use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use derive_getters::Getters;
use std::collections::BTreeMap;
use thiserror::Error;
use wasm_bindgen::{prelude::wasm_bindgen, JsError};

#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to parse vocabulary: {0}")]
    FailedToParseVocabulary(serde_json::Error),
    #[error("no matching token found")]
    NoMatchingTokenFound,
    #[error("out of range token: {0}")]
    OutOfRangeToken(u32),
}

#[derive(Debug, Clone, Getters)]
pub struct Tokenizer {
    first_bytes_to_lengths: Vec<Box<[u16]>>,
    bytes_to_token_index: HashMap<Vec<u8>, u32>,
    token_index_to_bytes: Vec<Vec<u8>>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum StrOrBytes {
    Str(String),
    Bytes(Vec<u8>),
}

impl Tokenizer {
    pub fn new(vocab: &str) -> Result<Self, TokenizerError> {
        let map: BTreeMap<u32, StrOrBytes> =
            serde_json::from_str(vocab).map_err(TokenizerError::FailedToParseVocabulary)?;

        let list: Vec<(Vec<u8>, u32)> = map
            .into_iter()
            .map(|(token, pattern)| {
                let pattern = match pattern {
                    StrOrBytes::Str(string) => string.into_bytes(),
                    StrOrBytes::Bytes(bytes) => bytes,
                };
                (pattern, token)
            })
            .collect();

        let mut first_bytes_to_len = Vec::new();
        first_bytes_to_len.resize(u16::MAX as usize, 2);

        let mut first_bytes_to_lengths = Vec::new();
        first_bytes_to_lengths.resize(u16::MAX as usize, {
            let mut set = HashSet::new();
            set.insert(1);
            set
        });

        let mut token_index_to_bytes = Vec::new();
        // Find the max token index to determine the size of the vector.
        let max_token_index = list.iter().map(|(_, index)| *index).max().unwrap_or(0) as usize;
        token_index_to_bytes.resize_with(max_token_index + 1, Vec::new);

        let mut bytes_to_token_index = HashMap::new();
        for (token_bytes, token_index) in list {
            if token_bytes.len() >= 2 {
                let key = u16::from_ne_bytes([token_bytes[0], token_bytes[1]]) as usize;
                let max_length = &mut first_bytes_to_len[key];
                if token_bytes.len() > *max_length {
                    *max_length = token_bytes.len();
                }

                first_bytes_to_lengths[key].insert(token_bytes.len() as u16);
            }

            bytes_to_token_index.insert(token_bytes.clone(), token_index);
            token_index_to_bytes[token_index as usize] = token_bytes;
        }

        let first_bytes_to_lengths: Vec<Box<[_]>> = first_bytes_to_lengths
            .into_iter()
            .map(|inner| {
                let mut inner: Vec<_> = inner.into_iter().collect();
                inner.sort_unstable_by_key(|l| !*l);
                inner.into_boxed_slice()
            })
            .collect();

        Ok(Tokenizer {
            first_bytes_to_lengths,
            bytes_to_token_index,
            token_index_to_bytes,
        })
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let mut output = Vec::new();
        self.encode_into(input, &mut output)?;
        Ok(output)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::with_capacity(tokens.len());
        self.decode_into(tokens, &mut output)?;
        Ok(output)
    }
}

impl Tokenizer {
    pub fn encode_into(
        &self,
        mut input: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        'next_token: while !input.is_empty() {
            let lengths = if input.len() >= 2 {
                let key = u16::from_ne_bytes([input[0], input[1]]) as usize;
                &self.first_bytes_to_lengths[key][..]
            } else {
                &[1][..]
            };

            for &length in lengths {
                let length = length as usize;
                if length > input.len() {
                    continue;
                }

                if let Some(&token_index) = self.bytes_to_token_index.get(&input[..length]) {
                    output.push(token_index);
                    input = &input[length..];
                    continue 'next_token;
                }
            }

            return Err(TokenizerError::NoMatchingTokenFound);
        }

        Ok(())
    }

    pub fn decode_into(&self, tokens: &[u32], output: &mut Vec<u8>) -> Result<(), TokenizerError> {
        for &token in tokens {
            let bytes = self
                .token_index_to_bytes
                .get(token as usize)
                .ok_or(TokenizerError::OutOfRangeToken(token))?;

            output.extend_from_slice(bytes);
        }

        Ok(())
    }
}

// Private types for HuggingFace tokenizer.json deserialization.
#[derive(Clone, serde::Deserialize)]
struct HfPreTokenizer {
    #[serde(default)]
    pattern: Option<HfPatternObj>,
    #[serde(default)]
    pretokenizers: Option<Vec<HfPreTokenizer>>,
}

#[derive(Clone, serde::Deserialize)]
struct HfPatternObj {
    #[serde(rename = "Regex", default)]
    regex: Option<String>,
}

/// A BPE tokenizer compatible with HuggingFace `tokenizer.json` format.
///
/// Used by Brumby (Qwen3-based) models. Supports the standard BPE encoding
/// algorithm with byte-level fallback and a pre-tokenization regex.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token bytes -> token ID.
    encoder: HashMap<Vec<u8>, u32>,
    /// Token ID -> token bytes.
    decoder: Vec<Vec<u8>>,
    /// Merge priority: (left, right) -> rank (lower = higher priority).
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    /// Pre-tokenization regex pattern.
    pat: regex::Regex,
}

impl BpeTokenizer {
    /// Construct a BPE tokenizer from a HuggingFace `tokenizer.json` file.
    pub fn new(tokenizer_json: &str) -> Result<Self, TokenizerError> {
        #[derive(serde::Deserialize)]
        struct TokenizerJson {
            model: ModelJson,
            #[serde(default)]
            added_tokens: Vec<AddedToken>,
            #[serde(default)]
            pre_tokenizer: Option<HfPreTokenizer>,
        }

        #[derive(serde::Deserialize)]
        struct ModelJson {
            vocab: std::collections::HashMap<String, u32>,
            #[serde(deserialize_with = "deserialize_merges")]
            merges: Vec<String>,
        }

        /// Accept merges as either `["a", "b"]` arrays or `"a b"` strings.
        fn deserialize_merges<'de, D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> Result<Vec<String>, D::Error> {
            #[derive(serde::Deserialize)]
            #[serde(untagged)]
            enum MergeEntry {
                Str(String),
                Pair([String; 2]),
            }
            let entries: Vec<MergeEntry> = serde::Deserialize::deserialize(deserializer)?;
            Ok(entries
                .into_iter()
                .map(|e| match e {
                    MergeEntry::Str(s) => s,
                    MergeEntry::Pair([a, b]) => format!("{a} {b}"),
                })
                .collect())
        }

        #[derive(serde::Deserialize)]
        struct AddedToken {
            id: u32,
            content: String,
        }

        let parsed: TokenizerJson = serde_json::from_str(tokenizer_json)
            .map_err(TokenizerError::FailedToParseVocabulary)?;

        // Build encoder map. Vocab keys use HuggingFace's byte-level encoding
        // where each byte is mapped to a printable unicode character.
        let mut encoder = HashMap::new();
        let mut max_id: u32 = 0;
        for (token_str, id) in &parsed.model.vocab {
            let bytes = Self::hf_decode_token(token_str);
            encoder.insert(bytes, *id);
            max_id = max_id.max(*id);
        }

        // Add special/added tokens.
        for at in &parsed.added_tokens {
            let bytes = at.content.as_bytes().to_vec();
            encoder.insert(bytes, at.id);
            max_id = max_id.max(at.id);
        }

        // Build decoder (id -> bytes).
        let mut decoder = vec![vec![]; (max_id + 1) as usize];
        for (bytes, &id) in &encoder {
            decoder[id as usize] = bytes.clone();
        }

        // Build merge ranks.
        let mut merge_ranks = HashMap::new();
        for (rank, merge_str) in parsed.model.merges.iter().enumerate() {
            if let Some((left, right)) = merge_str.split_once(' ') {
                let left = Self::hf_decode_token(left);
                let right = Self::hf_decode_token(right);
                merge_ranks.insert((left, right), rank);
            }
        }

        // Extract pre-tokenization regex pattern.
        // Note: Rust's `regex` crate does not support lookaheads (e.g. `(?!\S)`).
        // We strip the unsupported `\s+(?!\S)|` prefix from the trailing whitespace
        // alternatives, leaving just `\s+` which is functionally equivalent.
        let pat_str = Self::extract_pattern(&parsed.pre_tokenizer).unwrap_or_else(|| {
            // Default Qwen3/GPT-4 pattern for byte-level BPE (without lookahead).
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+".to_string()
        });
        // Strip lookahead constructs that Rust's regex crate doesn't support.
        let pat_str = Self::strip_lookaheads(&pat_str);
        let pat = regex::Regex::new(&pat_str).map_err(|_| TokenizerError::NoMatchingTokenFound)?;

        Ok(Self {
            encoder,
            decoder,
            merge_ranks,
            pat,
        })
    }

    /// Decode a HuggingFace byte-level BPE token string to raw bytes.
    fn hf_decode_token(token: &str) -> Vec<u8> {
        token.chars().map(Self::hf_char_to_byte).collect()
    }

    /// Reverse the HuggingFace byte-to-unicode mapping.
    fn hf_char_to_byte(c: char) -> u8 {
        let cp = c as u32;
        // HF byte-level BPE maps:
        //   bytes 33..=126 -> same codepoint (printable ASCII)
        //   bytes 161..=172, 174..=255 -> same codepoint (Latin-1)
        //   remaining 68 bytes (0..=32, 127..=160, 173) -> U+0100..U+0143
        match cp {
            33..=126 | 161..=172 | 174..=255 => cp as u8,
            _ => {
                let idx = cp - 256;
                if idx <= 32 {
                    idx as u8
                } else if idx <= 66 {
                    (idx - 33 + 127) as u8
                } else {
                    173u8
                }
            }
        }
    }

    /// Extract regex pattern from the pre_tokenizer JSON structure.
    fn extract_pattern(pre_tok: &Option<HfPreTokenizer>) -> Option<String> {
        let pt = pre_tok.as_ref()?;
        if let Some(pat) = &pt.pattern {
            if let Some(regex) = &pat.regex {
                return Some(regex.clone());
            }
        }
        if let Some(pretoks) = &pt.pretokenizers {
            for sub in pretoks {
                if let Some(pat) = Self::extract_pattern(&Some(sub.clone())) {
                    return Some(pat);
                }
            }
        }
        None
    }

    /// Strip lookahead/lookbehind constructs that Rust's `regex` crate doesn't support.
    /// Replaces `\s+(?!\S)|` with nothing (the subsequent `\s+` handles it).
    fn strip_lookaheads(pat: &str) -> String {
        pat.replace(r"\s+(?!\S)|", "").replace(r"\s+(?!\s)|", "")
    }

    /// Encode a string to token IDs using BPE.
    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        let text = String::from_utf8_lossy(input);
        let mut output = Vec::new();

        for mat in self.pat.find_iter(&text) {
            let piece = mat.as_str().as_bytes();
            self.bpe_encode_piece(piece, &mut output)?;
        }

        Ok(output)
    }

    /// Decode token IDs back to bytes.
    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut output = Vec::new();
        for &token in tokens {
            let bytes = self
                .decoder
                .get(token as usize)
                .ok_or(TokenizerError::OutOfRangeToken(token))?;
            output.extend_from_slice(bytes);
        }
        Ok(output)
    }

    /// BPE encode a single pre-tokenized piece.
    fn bpe_encode_piece(
        &self,
        piece: &[u8],
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        if piece.is_empty() {
            return Ok(());
        }

        // Start with individual bytes as tokens.
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();

        // Iteratively merge the highest-priority (lowest rank) pair.
        loop {
            if parts.len() < 2 {
                break;
            }

            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            for i in 0..parts.len() - 1 {
                if let Some(&rank) =
                    self.merge_ranks.get(&(parts[i].clone(), parts[i + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break;
            }

            let right = parts.remove(best_idx + 1);
            parts[best_idx].extend_from_slice(&right);
        }

        // Map merged byte sequences to token IDs.
        for part in &parts {
            if let Some(&id) = self.encoder.get(part) {
                output.push(id);
            } else {
                for &b in part {
                    if let Some(&id) = self.encoder.get(&vec![b]) {
                        output.push(id);
                    } else {
                        return Err(TokenizerError::NoMatchingTokenFound);
                    }
                }
            }
        }

        Ok(())
    }
}

#[wasm_bindgen(js_name = Tokenizer)]
pub struct JsTokenizer(Tokenizer);

#[wasm_bindgen(js_class = Tokenizer)]
impl JsTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(vocab: &str) -> Result<Self, JsError> {
        Ok(Self(Tokenizer::new(vocab)?))
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, JsError> {
        Ok(self.0.encode(input)?)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, JsError> {
        Ok(self.0.decode(tokens)?)
    }
}

#[wasm_bindgen(js_name = BpeTokenizer)]
pub struct JsBpeTokenizer(BpeTokenizer);

#[wasm_bindgen(js_class = BpeTokenizer)]
impl JsBpeTokenizer {
    #[wasm_bindgen(constructor)]
    pub fn new(tokenizer_json: &str) -> Result<Self, JsError> {
        Ok(Self(BpeTokenizer::new(tokenizer_json)?))
    }

    pub fn encode(&self, input: &[u8]) -> Result<Vec<u32>, JsError> {
        Ok(self.0.encode(input)?)
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<u8>, JsError> {
        Ok(self.0.decode(tokens)?)
    }
}
