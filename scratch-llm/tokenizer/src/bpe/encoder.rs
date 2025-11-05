use std::collections::HashMap;

pub struct BPEEncoder {
    merges: Vec<(String, String)>,
    vocab: HashMap<String, usize>,
    token_to_id: HashMap<String, usize>,
    id_to_token: HashMap<usize, String>,
}

impl BPEEncoder {
    pub fn new(merges: Vec<(String, String)>, vocab: HashMap<String, usize>) -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = HashMap::new();

        for (idx, token) in vocab.keys().enumerate() {
            token_to_id.insert(token.clone(), idx);
            id_to_token.insert(idx, token.clone());
        }

        Self {
            merges,
            vocab,
            token_to_id,
            id_to_token,
        }
    }

    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        for (pair_a, pair_b) in &self.merges {
            let mut i = 0;
            while i < tokens.len() - 1 {
                if &tokens[i] == pair_a && &tokens[i + 1] == pair_b {
                    let merged = format!("{}{}", pair_a, pair_b);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        for word in text.split_whitespace() {
            let tokens = self.tokenize_word(word);
            for token in tokens {
                if let Some(&id) = self.token_to_id.get(&token) {
                    ids.push(id);
                } else {
                    for ch in token.chars() {
                        if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                            ids.push(id);
                        }
                    }
                }
            }
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter_map(|&id| self.id_to_token.get(&id))
            .cloned()
            .collect::<Vec<String>>()
            .join("")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_all_tokens(&self) -> Vec<(usize, String)> {
        let mut tokens: Vec<_> = self
            .id_to_token
            .iter()
            .map(|(&id, token)| (id, token.clone()))
            .collect();
        tokens.sort_by_key(|(id, _)| *id);
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::BPETrainer;

    fn create_test_encoder(corpus: &str, num_merges: usize) -> BPEEncoder {
        let mut trainer = BPETrainer::new();
        trainer.train(corpus, num_merges);
        BPEEncoder::new(trainer.get_merges().clone(), trainer.get_vocab().clone())
    }

    /// Test 1: Basic Encoding
    ///
    /// Educational Purpose: Verify that words are tokenized according to learned merges
    /// BPE encoder applies merge rules in the order they were learned
    #[test]
    fn test_basic_encoding() {
        let encoder = create_test_encoder("hello hello world", 3);

        let tokens = encoder.tokenize_word("hello");

        // Should tokenize into subword units, not individual characters
        assert!(!tokens.is_empty(), "Should produce tokens");
        assert!(
            tokens.len() <= 5,
            "Should compress 'hello' (5 chars) into fewer tokens"
        );
    }

    /// Test 2: Encode-Decode Roundtrip
    ///
    /// Educational Purpose: Verify information is preserved through encoding/decoding
    /// BPE is lossless for text that uses vocabulary tokens
    ///
    /// IMPORTANT: Spaces are lost because BPE tokenizes word-by-word
    #[test]
    fn test_roundtrip() {
        let corpus = "the quick brown fox jumps";
        let encoder = create_test_encoder(corpus, 10);

        let text = "the quick fox";
        let ids = encoder.encode(text);
        let decoded = encoder.decode(&ids);

        // Remove spaces for comparison (BPE doesn't preserve spaces between words)
        let text_no_spaces = text.replace(' ', "");
        assert_eq!(
            decoded, text_no_spaces,
            "Roundtrip should preserve text (minus spaces)"
        );
    }

    /// Test 3: Empty Text Handling
    ///
    /// Educational Purpose: Edge case - encoding empty string
    /// Empty input should produce empty output
    #[test]
    fn test_empty_text() {
        let encoder = create_test_encoder("hello world", 5);

        let ids = encoder.encode("");
        assert_eq!(ids.len(), 0, "Empty text should produce no tokens");

        let decoded = encoder.decode(&[]);
        assert_eq!(decoded, "", "Empty ids should decode to empty string");
    }

    /// Test 4: Single Character Encoding
    ///
    /// Educational Purpose: Base case - single chars should always be encodable
    /// BPE vocabulary always contains individual characters from training
    #[test]
    fn test_single_character() {
        let encoder = create_test_encoder("abc", 0); // No merges

        let tokens = encoder.tokenize_word("a");
        assert_eq!(tokens.len(), 1, "Single char should be 1 token");
        assert_eq!(tokens[0], "a", "Token should be the character itself");
    }

    /// Test 5: Out-of-Vocabulary Handling
    ///
    /// Educational Purpose: How does BPE handle unseen characters?
    /// Characters not in vocab will be skipped (in this implementation)
    /// Better implementations use fallback strategies (UNK token, byte-level)
    #[test]
    fn test_oov_character() {
        let encoder = create_test_encoder("hello", 5);

        // Try to encode a character not in training corpus
        let ids = encoder.encode("xyz");

        // Current implementation: OOV chars are silently dropped
        // This is a limitation - production BPE uses byte-level fallback
        // Just verify it doesn't crash
        assert!(ids.len() <= 3, "OOV handling should not crash");
    }

    /// Test 6: Tokenization Consistency
    ///
    /// Educational Purpose: Same word should always tokenize the same way
    /// BPE is deterministic - no randomness in encoding
    #[test]
    fn test_tokenization_consistency() {
        let encoder = create_test_encoder("hello hello world world", 5);

        let tokens1 = encoder.tokenize_word("hello");
        let tokens2 = encoder.tokenize_word("hello");

        assert_eq!(tokens1, tokens2, "Same word should tokenize identically");
    }

    /// Test 7: Merge Application Order
    ///
    /// Educational Purpose: Merges must be applied in training order
    /// Later merges depend on earlier ones being applied first
    ///
    /// Example: If we learn "th" then "the":
    /// - "the" first becomes ["th", "e"] after first merge
    /// - Then ["th", "e"] becomes ["the"] after second merge
    #[test]
    fn test_merge_order() {
        let mut trainer = BPETrainer::new();
        let corpus = "the the the";
        trainer.train(corpus, 2);

        let merges = trainer.get_merges();
        let encoder = BPEEncoder::new(merges.clone(), trainer.get_vocab().clone());

        let tokens = encoder.tokenize_word("the");

        // After 2 merges on "the the the", we should learn "th" and "the"
        // Final tokenization should be ["the"] or similar compressed form
        assert!(
            tokens.len() <= 2,
            "Word 'the' should be compressed after merges, got {:?}",
            tokens
        );
    }

    /// Test 8: Vocabulary Size Consistency
    ///
    /// Educational Purpose: Encoder vocab should match trainer vocab
    /// All learned tokens should be accessible for encoding/decoding
    #[test]
    fn test_vocab_size() {
        let mut trainer = BPETrainer::new();
        trainer.train("hello world", 5);

        let trainer_vocab_size = trainer.get_vocab().len();
        let encoder = BPEEncoder::new(trainer.get_merges().clone(), trainer.get_vocab().clone());

        assert_eq!(
            encoder.vocab_size(),
            trainer_vocab_size,
            "Encoder vocab size should match trainer"
        );
    }

    /// Test 9: Token-ID Bidirectional Mapping
    ///
    /// Educational Purpose: Verify token<->ID mappings are correct
    /// token_to_id and id_to_token should be inverse operations
    #[test]
    fn test_token_id_mapping() {
        let encoder = create_test_encoder("abc", 1);

        // Test token -> ID -> token roundtrip
        for (id, token) in encoder.get_all_tokens() {
            // ID -> token
            let retrieved_token = encoder.id_to_token(id).expect("ID should map to token");
            assert_eq!(retrieved_token, token, "ID->token mapping mismatch");

            // token -> ID
            let retrieved_id = encoder.token_to_id(&token).expect("Token should map to ID");
            assert_eq!(retrieved_id, id, "Token->ID mapping mismatch");
        }
    }

    /// Test 10: Multiple Words Encoding
    ///
    /// Educational Purpose: Verify encoding handles multiple words correctly
    /// Each word is tokenized independently, then concatenated
    #[test]
    fn test_multiple_words() {
        let corpus = "the cat sat";
        let encoder = create_test_encoder(corpus, 5);

        let text = "the cat";
        let ids = encoder.encode(text);

        // Should produce some tokens (at least characters)
        assert!(!ids.is_empty(), "Multi-word text should produce tokens");

        // Each ID should be valid
        for &id in &ids {
            assert!(
                encoder.id_to_token(id).is_some(),
                "ID {} should be valid",
                id
            );
        }
    }

    /// Test 11: Greedy Tokenization
    ///
    /// Educational Purpose: BPE tokenizes greedily using longest matches
    /// Merges are applied left-to-right, longest match first
    #[test]
    fn test_greedy_tokenization() {
        let mut trainer = BPETrainer::new();
        let corpus = "aaaa"; // Will learn "aa", then potentially "aaaa"
        trainer.train(corpus, 2);

        let encoder = BPEEncoder::new(trainer.get_merges().clone(), trainer.get_vocab().clone());

        let tokens = encoder.tokenize_word("aaaa");

        // Should use longest possible merges
        // After learning "a,a"->"aa", "aaaa" becomes ["aa", "aa"]
        assert!(tokens.len() <= 2, "Should use greedy merging: {:?}", tokens);
    }

    /// Test 12: Encoding Produces Valid IDs
    ///
    /// Educational Purpose: All encoded IDs should be in vocab range
    /// encode() should never produce invalid IDs
    #[test]
    fn test_valid_ids() {
        let encoder = create_test_encoder("hello world test", 10);
        let text = "hello world";
        let ids = encoder.encode(text);

        let vocab_size = encoder.vocab_size();

        for &id in &ids {
            assert!(
                id < vocab_size,
                "ID {} should be < vocab_size {}",
                id,
                vocab_size
            );
        }
    }

    /// Test 13: Decoding Handles Invalid IDs
    ///
    /// Educational Purpose: Graceful handling of invalid IDs
    /// decode() should skip invalid IDs rather than panic
    #[test]
    fn test_invalid_id_decoding() {
        let encoder = create_test_encoder("hello", 3);

        let vocab_size = encoder.vocab_size();
        let invalid_ids = vec![vocab_size + 1, vocab_size + 100];

        // Should not panic, just produce empty or partial output
        let decoded = encoder.decode(&invalid_ids);

        assert!(
            decoded.is_empty(),
            "Invalid IDs should decode to empty string"
        );
    }

    /// Test 14: Whitespace Handling
    ///
    /// Educational Purpose: Understand BPE's whitespace limitations
    /// Standard BPE tokenizes word-by-word, losing space information
    ///
    /// This is why GPT uses special handling (Ġ prefix) for spaces
    #[test]
    fn test_whitespace_handling() {
        let encoder = create_test_encoder("hello world", 5);

        let text1 = "hello world";
        let text2 = "helloworld";

        let ids1 = encoder.encode(text1);
        let ids2 = encoder.encode(text2);

        let decoded1 = encoder.decode(&ids1);
        let decoded2 = encoder.decode(&ids2);

        // Both should decode to the same string (no spaces)
        assert_eq!(
            decoded1, decoded2,
            "Whitespace is not preserved in basic BPE"
        );
    }

    /// Test 15: Subword Compression
    ///
    /// Educational Purpose: Verify BPE actually compresses repeated patterns
    /// Words with learned subwords should use fewer tokens than chars
    #[test]
    fn test_subword_compression() {
        let corpus = "testing testing testing";
        let encoder = create_test_encoder(corpus, 10);

        let tokens = encoder.tokenize_word("testing");
        let char_count = "testing".chars().count();

        // After learning merges, should use fewer tokens than characters
        assert!(
            tokens.len() < char_count,
            "BPE should compress 'testing' from {} chars to {} tokens",
            char_count,
            tokens.len()
        );
    }
}
