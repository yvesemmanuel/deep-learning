use log::{debug, info};
use std::collections::HashMap;

/// Training metrics collected during BPE training process
///
/// These metrics help us understand and visualize how the BPE algorithm
/// progressively learns subword units from the corpus.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Vocabulary size after each merge iteration
    /// Tracks how the vocabulary grows as we merge frequent pairs
    pub vocab_sizes: Vec<usize>,

    /// Total number of tokens in the corpus after each merge
    /// Shows compression: fewer tokens = better compression
    pub token_counts: Vec<usize>,

    /// The merged pairs with their frequencies at each step
    /// Records which pairs were most frequent and got merged
    pub merge_history: Vec<((String, String), usize)>,

    /// Number of unique words in the training corpus
    pub unique_words: usize,

    /// Total number of word occurrences
    pub total_words: usize,
}

impl TrainingMetrics {
    pub fn new(unique_words: usize, total_words: usize) -> Self {
        Self {
            vocab_sizes: Vec::new(),
            token_counts: Vec::new(),
            merge_history: Vec::new(),
            unique_words,
            total_words,
        }
    }

    /// Calculate compression ratio: initial tokens / current tokens
    /// Higher ratio = better compression
    pub fn compression_ratio(&self) -> f32 {
        if self.token_counts.is_empty() {
            return 1.0;
        }
        let initial = self.token_counts[0] as f32;
        let final_count = *self.token_counts.last().unwrap() as f32;
        initial / final_count
    }

    /// Calculate average tokens per merge reduction
    pub fn avg_token_reduction_per_merge(&self) -> f32 {
        if self.token_counts.len() < 2 {
            return 0.0;
        }
        let initial = self.token_counts[0] as f32;
        let final_count = *self.token_counts.last().unwrap() as f32;
        (initial - final_count) / (self.token_counts.len() - 1) as f32
    }
}

pub struct BPETrainer {
    pub vocab: HashMap<String, usize>,
    pub merges: Vec<(String, String)>,
    pub metrics: Option<TrainingMetrics>,
}

impl BPETrainer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
            metrics: None,
        }
    }

    /// Train the BPE tokenizer on a corpus
    ///
    /// BPE Algorithm Overview:
    /// 1. Start by splitting all words into individual characters
    /// 2. Count frequency of all adjacent character pairs
    /// 3. Merge the most frequent pair into a new token
    /// 4. Repeat steps 2-3 for num_merges iterations
    ///
    /// This greedy algorithm learns subword units that balance between:
    /// - Character-level (high granularity, long sequences)
    /// - Word-level (compact but huge vocabulary)
    pub fn train(&mut self, corpus: &str, num_merges: usize) {
        // Step 1: Count word frequencies in corpus
        let mut word_freqs = HashMap::new();
        for word in corpus.split_whitespace() {
            *word_freqs.entry(word.to_string()).or_insert(0) += 1;
        }

        let unique_words = word_freqs.len();
        let total_words: usize = word_freqs.values().sum();

        let mut metrics = TrainingMetrics::new(unique_words, total_words);

        // Step 2: Split each word into character tokens
        // This is the initial representation: fully character-level
        let mut splits: HashMap<String, Vec<String>> = word_freqs
            .keys()
            .map(|word| {
                let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                (word.clone(), chars)
            })
            .collect();

        // Step 3: Build initial vocabulary from all unique characters
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for split in splits.values() {
            for token in split {
                *vocab.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let initial_token_count = self.count_total_tokens(&splits, &word_freqs);
        metrics.vocab_sizes.push(vocab.len());
        metrics.token_counts.push(initial_token_count);

        info!("Initial vocabulary size: {}", vocab.len());
        info!("Initial token count: {}", initial_token_count);
        info!("Training BPE with {} merges...", num_merges);

        // Step 4: Iteratively merge the most frequent pair
        for merge_idx in 0..num_merges {
            let pair_freqs = self.compute_pair_frequencies(&splits, &word_freqs);

            if pair_freqs.is_empty() {
                info!("No more pairs to merge at iteration {}", merge_idx);
                break;
            }

            // Select the pair with highest frequency, the greedy choice
            let (best_pair, best_freq) = pair_freqs
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(pair, &freq)| (pair.clone(), freq))
                .unwrap();

            debug!(
                "Merge {}: {:?} -> {} (freq: {})",
                merge_idx + 1,
                best_pair,
                format!("{}{}", best_pair.0, best_pair.1),
                best_freq
            );

            self.merge_pair(&mut splits, &best_pair);

            // Update vocabulary
            self.merges.push(best_pair.clone());
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            *vocab.entry(new_token).or_insert(0) += 1;

            // Record metrics after merge
            let current_token_count = self.count_total_tokens(&splits, &word_freqs);
            metrics.vocab_sizes.push(vocab.len());
            metrics.token_counts.push(current_token_count);
            metrics.merge_history.push((best_pair, best_freq));
        }

        self.vocab = vocab;
        self.metrics = Some(metrics);

        info!("Final vocabulary size: {}", self.vocab.len());
        info!(
            "Token count reduction: {} -> {}",
            self.metrics.as_ref().unwrap().token_counts[0],
            self.metrics.as_ref().unwrap().token_counts.last().unwrap()
        );
    }

    /// Count total tokens across all words (weighted by frequency)
    /// This metric shows how well we're compressing the corpus
    fn count_total_tokens(
        &self,
        splits: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, usize>,
    ) -> usize {
        let mut total = 0;
        for (word, split) in splits {
            let freq = word_freqs.get(word).unwrap_or(&0);
            total += split.len() * freq;
        }
        total
    }

    /// Compute frequency of all adjacent token pairs in the corpus
    ///
    /// BPE uses a greedy approach - always merge the most frequent pair.
    /// This is a frequency-based compression algorithm similar to Huffman coding.
    fn compute_pair_frequencies(
        &self,
        splits: &HashMap<String, Vec<String>>,
        word_freqs: &HashMap<String, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_freqs = HashMap::new();

        for (word, freq) in word_freqs {
            let split = &splits[word];
            if split.len() < 2 {
                continue;
            }

            // Count each adjacent pair, weighted by word frequency
            for i in 0..split.len() - 1 {
                let pair = (split[i].clone(), split[i + 1].clone());
                *pair_freqs.entry(pair).or_insert(0) += freq;
            }
        }

        pair_freqs
    }

    /// Apply a merge operation to all word splits
    ///
    /// Replaces every occurrence of (pair.0, pair.1) with the merged token
    /// This is applied consistently across all words in the vocabulary
    fn merge_pair(&self, splits: &mut HashMap<String, Vec<String>>, pair: &(String, String)) {
        for split in splits.values_mut() {
            let mut i = 0;
            while i < split.len() - 1 {
                if split[i] == pair.0 && split[i + 1] == pair.1 {
                    // Merge the pair into a single token
                    let merged = format!("{}{}", pair.0, pair.1);
                    split[i] = merged;
                    split.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    pub fn get_vocab(&self) -> &HashMap<String, usize> {
        &self.vocab
    }

    pub fn get_merges(&self) -> &Vec<(String, String)> {
        &self.merges
    }

    pub fn get_metrics(&self) -> Option<&TrainingMetrics> {
        self.metrics.as_ref()
    }
}

impl Default for BPETrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test 1: Basic BPE Training
    ///
    /// Understand how BPE builds vocabulary from scratch
    /// BPE starts with character-level tokens and progressively merges
    /// the most frequent pairs to learn subword units.
    #[test]
    fn test_basic_bpe_training() {
        let mut trainer = BPETrainer::new();
        let corpus = "the the the cat";

        trainer.train(corpus, 2);

        assert_eq!(trainer.merges.len(), 2, "Should perform exactly 2 merges");

        let vocab = trainer.get_vocab();
        assert!(!vocab.is_empty(), "Vocabulary should not be empty");

        assert!(vocab.contains_key("t"), "Should contain character 't'");
        assert!(vocab.contains_key("h"), "Should contain character 'h'");
        assert!(vocab.contains_key("e"), "Should contain character 'e'");
    }

    /// Test 2: Merge Order and Priority
    ///
    /// BPE is greedy - it always merges the MOST frequent pair
    /// This greedy approach ensures maximum compression at each step
    ///
    /// Expected Behavior: Pairs from "the" should merge since it appears 3x vs "cat" 1x
    #[test]
    fn test_merge_order() {
        let mut trainer = BPETrainer::new();
        // "the" appears 3 times, so pairs from "the" have higher frequency
        // "cat" appears 1 time, so pairs from "cat" have lower frequency
        let corpus = "the the the cat";

        trainer.train(corpus, 3);

        let merges = trainer.get_merges();
        assert!(!merges.is_empty(), "Should have performed merges");

        // Get frequencies from metrics to verify greedy property
        if let Some(metrics) = trainer.get_metrics() {
            let merge_freqs: Vec<usize> = metrics
                .merge_history
                .iter()
                .map(|(_, freq)| *freq)
                .collect();

            // Verify frequencies are non-increasing (greedy property)
            for i in 1..merge_freqs.len() {
                assert!(
                    merge_freqs[i] <= merge_freqs[i - 1],
                    "Frequencies should be non-increasing: {} > {} at position {}",
                    merge_freqs[i],
                    merge_freqs[i - 1],
                    i
                );
            }

            // First merge should have highest frequency
            if !merge_freqs.is_empty() {
                let max_freq = merge_freqs.iter().max().unwrap();
                assert_eq!(
                    merge_freqs[0], *max_freq,
                    "First merge should have maximum frequency"
                );
            }
        }
    }

    /// Test 3: Vocabulary Growth Pattern
    ///
    /// Understand how vocabulary size increases with merges
    /// Each merge adds one new token to vocabulary (the merged pair)
    /// Base vocabulary stays constant (original characters)
    #[test]
    fn test_vocabulary_growth() {
        let mut trainer = BPETrainer::new();
        let corpus = "aaa bbb";
        let num_merges = 2;

        trainer.train(corpus, num_merges);

        let metrics = trainer.get_metrics().expect("Metrics should be available");

        assert_eq!(
            metrics.vocab_sizes.len(),
            1 + num_merges, // initial + num_merges
            "Should track vocab size at each step"
        );

        for i in 1..metrics.vocab_sizes.len() {
            assert!(
                metrics.vocab_sizes[i] >= metrics.vocab_sizes[i - 1],
                "Vocabulary should grow or stay constant, not shrink"
            );
        }
    }

    /// Test 4: Token Count Reduction (Compression)
    ///
    /// BPE compresses text by reducing token count
    /// Merging pairs reduces total tokens while preserving information
    ///
    /// Example: "aaa" = [a, a, a] (3 tokens) -> [aa, a] (2 tokens) -> [aaa] (1 token)
    #[test]
    fn test_token_compression() {
        let mut trainer = BPETrainer::new();
        let corpus = "aaa aaa aaa"; // Highly repetitive for maximum compression
        let num_merges = 2;

        trainer.train(corpus, num_merges);

        let metrics = trainer.get_metrics().expect("Metrics should be available");
        let token_counts = &metrics.token_counts;

        // Token count should decrease as we merge
        assert!(token_counts.len() >= 2, "Should track token counts");

        let initial_count = token_counts[0];
        let final_count = *token_counts.last().unwrap();

        assert!(
            final_count < initial_count,
            "Token count should decrease: {} -> {}",
            initial_count,
            final_count
        );

        // Compression ratio should be > 1
        let compression = metrics.compression_ratio();
        assert!(
            compression > 1.0,
            "Compression ratio should be > 1.0, got {}",
            compression
        );
    }

    /// Test 5: Empty and Edge Cases
    ///
    /// Ensure robust handling of edge cases
    /// BPE should gracefully handle empty inputs and single characters
    #[test]
    fn test_empty_corpus() {
        let mut trainer = BPETrainer::new();
        trainer.train("", 10);

        assert_eq!(
            trainer.get_vocab().len(),
            0,
            "Empty corpus should result in empty vocab"
        );
        assert_eq!(
            trainer.get_merges().len(),
            0,
            "Empty corpus should have no merges"
        );
    }

    /// Test 6: Single Character Words
    ///
    /// BPE needs at least 2 characters to form pairs
    /// Single character words cannot be merged
    #[test]
    fn test_single_char_words() {
        let mut trainer = BPETrainer::new();
        let corpus = "a b c d e"; // No pairs possible

        trainer.train(corpus, 10);

        // Should have no merges since no pairs exist
        assert_eq!(
            trainer.get_merges().len(),
            0,
            "Single character words should produce no merges"
        );
    }

    /// Test 7: Repeated Pattern Recognition
    ///
    /// BPE excels at learning repeated patterns
    /// High-frequency patterns get merged early, forming subword units
    ///
    /// Example: "tion" suffix in English appears frequently
    #[test]
    fn test_repeated_patterns() {
        let mut trainer = BPETrainer::new();
        // "ing" pattern repeats - should be learned
        let corpus = "running jumping singing dancing";

        trainer.train(corpus, 5);

        let merges = trainer.get_merges();

        // Check if any merge involves 'n' and 'g' (part of "ing")
        let has_ng_merge = merges
            .iter()
            .any(|(a, b)| (a == "n" && b == "g") || a.ends_with('n') && b.starts_with('g'));

        assert!(
            has_ng_merge,
            "Should learn common pattern 'ng' from repeated suffix"
        );
    }

    /// Test 8: Merge History Tracking
    ///
    /// Understanding which pairs are merged helps analyze
    /// what the model learns about language structure
    /// Merge history shows the priority order of subword units
    #[test]
    fn test_merge_history() {
        let mut trainer = BPETrainer::new();
        let corpus = "hello hello world";

        trainer.train(corpus, 3);

        let metrics = trainer.get_metrics().expect("Metrics should exist");

        assert_eq!(
            metrics.merge_history.len(),
            3,
            "Should record exactly 3 merges"
        );

        // Each merge should have a non-zero frequency
        for ((pair_a, pair_b), freq) in &metrics.merge_history {
            assert!(
                *freq > 0,
                "Merge ({}, {}) should have positive frequency",
                pair_a,
                pair_b
            );
        }

        // Frequencies should be non-increasing (greedy merges most frequent first)
        for i in 1..metrics.merge_history.len() {
            let prev_freq = metrics.merge_history[i - 1].1;
            let curr_freq = metrics.merge_history[i].1;
            assert!(
                curr_freq <= prev_freq,
                "Merge frequencies should be non-increasing (greedy): {} <= {}",
                curr_freq,
                prev_freq
            );
        }
    }

    /// Test 9: Consistency of Training Results
    ///
    /// BPE training with same corpus should produce consistent results
    /// Same corpus and num_merges should produce same vocab size and merge count
    ///
    /// Note: This implementation uses HashMap which has non-deterministic iteration order
    /// for security reasons. When multiple pairs have equal frequency, the selection order
    /// may vary between runs. Production BPE implementations use sorted data structures
    /// for true determinism. Here we test that the key properties remain consistent.
    /// TODO: implement a sorted data structure
    #[test]
    fn test_training_consistency() {
        let corpus = "aaaa bbbb cccc"; // Highly regular corpus
        let num_merges = 3;

        let mut trainer1 = BPETrainer::new();
        trainer1.train(corpus, num_merges);

        let mut trainer2 = BPETrainer::new();
        trainer2.train(corpus, num_merges);

        // 1. Vocabulary sizes should match
        assert_eq!(
            trainer1.get_vocab().len(),
            trainer2.get_vocab().len(),
            "Vocabulary sizes should be consistent"
        );

        // 2. Number of merges should match
        assert_eq!(
            trainer1.get_merges().len(),
            trainer2.get_merges().len(),
            "Number of merges should be consistent"
        );

        // 3. Metrics should be consistent
        let metrics1 = trainer1.get_metrics().unwrap();
        let metrics2 = trainer2.get_metrics().unwrap();

        assert_eq!(
            metrics1.vocab_sizes.len(),
            metrics2.vocab_sizes.len(),
            "Should track same number of vocab size measurements"
        );

        assert_eq!(
            metrics1.token_counts, metrics2.token_counts,
            "Token counts should be identical (compression is deterministic)"
        );

        // 4. Compression ratio should be identical
        assert_eq!(
            metrics1.compression_ratio(),
            metrics2.compression_ratio(),
            "Compression ratio should be consistent"
        );
    }

    /// Test 10: Unicode Support
    ///
    /// BPE should handle unicode characters correctly
    /// Character-level splitting should work with any unicode text
    #[test]
    fn test_unicode_support() {
        let mut trainer = BPETrainer::new();
        let corpus = "hello world こんにちは 你好";

        trainer.train(corpus, 5);

        let vocab = trainer.get_vocab();

        // Should contain unicode characters
        assert!(!vocab.is_empty(), "Should handle unicode corpus");

        assert!(
            trainer.get_merges().len() <= 5,
            "Should complete training without errors"
        );
    }
}
