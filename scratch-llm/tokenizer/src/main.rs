mod bpe;
mod embeddings;
mod viz;

use bpe::{BPEEncoder, BPETrainer};
use embeddings::{EmbeddingComparison, LearnedEmbedding, OneHotEmbedding};
use log::{error, info};
use viz::plotter;
use viz::TokenVisualizer;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("Tokenizer - BPE algorithm\n");

    info!("\nSTEP 1: Training Byte-Pair Encoder");
    info!("{}", "=".repeat(60));

    let corpus = "the quick brown fox jumps over the lazy dog \
                  the dog barks at the fox \
                  quick thinking saves the day \
                  the lazy cat sleeps all day";

    let mut trainer = BPETrainer::new();
    trainer.train(corpus, 15);

    let vocab = trainer.get_vocab().clone();
    let merges = trainer.get_merges().clone();

    if let Some(metrics) = trainer.get_metrics() {
        info!("\nTraining Statistics:");
        info!("  Compression ratio: {:.2}x", metrics.compression_ratio());
        info!(
            "  Avg token reduction per merge: {:.2}",
            metrics.avg_token_reduction_per_merge()
        );
        info!("  Unique words in corpus: {}", metrics.unique_words);
        info!("  Total word occurrences: {}", metrics.total_words);

        match plotter::plot_training_metrics(metrics, "bpe_training_metrics.png") {
            Ok(_) => info!("Training metrics visualization saved!"),
            Err(e) => error!("Error creating training metrics plot: {}", e),
        }
    }

    info!("\n\nSTEP 2: Token Visualization");
    info!("{}", "=".repeat(60));

    let encoder = BPEEncoder::new(merges, vocab);
    let visualizer = TokenVisualizer::new(encoder);

    visualizer.show_vocab_stats();

    let test_text = "the quick fox";
    visualizer.visualize_text(test_text);

    let words = vec!["the", "quick", "fox", "jumps", "lazy", "thinking"];
    visualizer.display_word_mapping(&words);

    visualizer.visualize_roundtrip("the lazy dog jumps");

    info!("\n\nSTEP 3: Embedding Comparison");
    info!("{}", "=".repeat(60));

    let mut trainer2 = BPETrainer::new();
    trainer2.train(corpus, 10);
    let encoder2 = BPEEncoder::new(trainer2.get_merges().clone(), trainer2.get_vocab().clone());

    let vocab_size = encoder2.vocab_size();
    info!("Vocabulary size: {}", vocab_size);

    let one_hot = OneHotEmbedding::new(vocab_size);
    let learned = LearnedEmbedding::new(vocab_size, 16);

    info!("\nOne-Hot embedding dimension: {}", one_hot.embedding_dim());
    info!("Learned embedding dimension: {}", learned.embedding_dim());

    let sample_tokens: Vec<usize> = (0..vocab_size.min(10)).collect();

    info!(
        "\nComparing embeddings for {} tokens...",
        sample_tokens.len()
    );

    let comparison = EmbeddingComparison::new(sample_tokens.clone(), &one_hot, &learned);
    comparison.display_stats();

    info!("\n\nSTEP 4: Plotting Cosine Distances");
    info!("{}", "=".repeat(60));

    match plotter::plot_distance_comparison(
        &comparison.one_hot_distances,
        &comparison.learned_distances,
        &comparison.token_ids,
        "distance_heatmaps.png",
    ) {
        Ok(_) => info!("Heatmap visualization saved!"),
        Err(e) => error!("Error creating heatmap: {}", e),
    }

    match plotter::plot_distance_histograms(
        &comparison.one_hot_distances,
        &comparison.learned_distances,
        "distance_histograms.png",
    ) {
        Ok(_) => info!("Histogram visualization saved!"),
        Err(e) => error!("Error creating histogram: {}", e),
    }
}
