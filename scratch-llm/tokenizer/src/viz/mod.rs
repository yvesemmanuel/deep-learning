pub mod plotter;

use crate::bpe::BPEEncoder;
use log::info;
use std::collections::HashMap;

pub struct TokenVisualizer {
    encoder: BPEEncoder,
}

impl TokenVisualizer {
    pub fn new(encoder: BPEEncoder) -> Self {
        Self { encoder }
    }

    pub fn visualize_text(&self, text: &str) {
        info!("\n{}", "=".repeat(60));
        info!("TOKEN VISUALIZATION");
        info!("{}", "=".repeat(60));
        info!("Input text: \"{}\"", text);
        info!("{}", "-".repeat(60));

        for word in text.split_whitespace() {
            let tokens = self.encoder.tokenize_word(word);
            let ids: Vec<usize> = tokens
                .iter()
                .filter_map(|t| self.encoder.token_to_id(t))
                .collect();

            info!("\nWord: '{}'", word);
            info!("  Tokens: {:?}", tokens);
            info!("  IDs:    {:?}", ids);

            let visual: String = tokens
                .iter()
                .zip(ids.iter())
                .map(|(token, id)| format!("[{}:{}] ", token, id))
                .collect();
            info!("  Visual: {}", visual);
        }

        info!("{}", "=".repeat(60));
    }

    pub fn show_vocab_stats(&self) {
        info!("\n{}", "=".repeat(60));
        info!("VOCABULARY STATISTICS");
        info!("{}", "=".repeat(60));
        info!("Total tokens: {}", self.encoder.vocab_size());

        let tokens = self.encoder.get_all_tokens();

        info!("\nFirst 20 tokens:");
        for (id, token) in tokens.iter().take(20) {
            info!("  ID {}: '{}'", id, token);
        }

        if tokens.len() > 20 {
            info!("  ... ({} more tokens)", tokens.len() - 20);
        }

        info!("{}", "=".repeat(60));
    }

    pub fn create_word_mapping(&self, words: &[&str]) -> HashMap<String, Vec<(String, usize)>> {
        let mut mapping = HashMap::new();

        for &word in words {
            let tokens = self.encoder.tokenize_word(word);
            let token_id_pairs: Vec<(String, usize)> = tokens
                .iter()
                .filter_map(|t| self.encoder.token_to_id(t).map(|id| (t.clone(), id)))
                .collect();
            mapping.insert(word.to_string(), token_id_pairs);
        }

        mapping
    }

    pub fn display_word_mapping(&self, words: &[&str]) {
        info!("\n{}", "=".repeat(60));
        info!("WORD -> TOKEN MAPPING TABLE");
        info!("{}", "=".repeat(60));

        let mapping = self.create_word_mapping(words);

        for word in words {
            if let Some(token_pairs) = mapping.get(*word) {
                info!("\n'{}' ->", word);
                for (token, id) in token_pairs {
                    info!("    '{}' (ID: {})", token, id);
                }
            }
        }

        info!("{}", "=".repeat(60));
    }

    pub fn visualize_roundtrip(&self, text: &str) {
        info!("\n{}", "=".repeat(60));
        info!("ENCODING/DECODING ROUND-TRIP");
        info!("{}", "=".repeat(60));

        info!("Original:  \"{}\"", text);

        let ids = self.encoder.encode(text);
        info!("Encoded:   {:?}", ids);

        let decoded = self.encoder.decode(&ids);
        info!("Decoded:   \"{}\"", decoded);

        let match_status = if text.replace(' ', "") == decoded {
            "SUCCESS (with spaces removed)"
        } else if text == decoded {
            "PERFECT MATCH"
        } else {
            "MISMATCH"
        };
        info!("Status:    {}", match_status);

        info!("{}", "=".repeat(60));
    }
}
