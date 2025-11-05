use log::info;
use rand::Rng;
use std::collections::HashMap;

pub struct OneHotEmbedding {
    vocab_size: usize,
}

impl OneHotEmbedding {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    pub fn embed(&self, token_id: usize) -> Vec<f32> {
        let mut vec = vec![0.0; self.vocab_size];
        if token_id < self.vocab_size {
            vec[token_id] = 1.0;
        }
        vec
    }

    pub fn embed_sequence(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids.iter().map(|&id| self.embed(id)).collect()
    }

    pub fn embedding_dim(&self) -> usize {
        self.vocab_size
    }
}

pub struct LearnedEmbedding {
    embeddings: HashMap<usize, Vec<f32>>,
    embedding_dim: usize,
}

impl LearnedEmbedding {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut embeddings = HashMap::new();

        for token_id in 0..vocab_size {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            embeddings.insert(token_id, embedding);
        }

        Self {
            embeddings,
            embedding_dim,
        }
    }

    pub fn embed(&self, token_id: usize) -> Option<Vec<f32>> {
        self.embeddings.get(&token_id).cloned()
    }

    pub fn embed_sequence(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids.iter().filter_map(|&id| self.embed(id)).collect()
    }

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if magnitude_a == 0.0 || magnitude_b == 0.0 {
        return 0.0;
    }

    dot_product / (magnitude_a * magnitude_b)
}

pub fn compute_pairwise_distances(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut distances = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            let similarity = cosine_similarity(&embeddings[i], &embeddings[j]);
            distances[i][j] = 1.0 - similarity;
        }
    }

    distances
}

pub struct EmbeddingComparison {
    pub one_hot_distances: Vec<Vec<f32>>,
    pub learned_distances: Vec<Vec<f32>>,
    pub token_ids: Vec<usize>,
}

impl EmbeddingComparison {
    pub fn new(
        token_ids: Vec<usize>,
        one_hot: &OneHotEmbedding,
        learned: &LearnedEmbedding,
    ) -> Self {
        let one_hot_embeds = one_hot.embed_sequence(&token_ids);
        let learned_embeds = learned.embed_sequence(&token_ids);

        let one_hot_distances = compute_pairwise_distances(&one_hot_embeds);
        let learned_distances = compute_pairwise_distances(&learned_embeds);

        Self {
            one_hot_distances,
            learned_distances,
            token_ids,
        }
    }

    pub fn display_stats(&self) {
        info!("\n{}", "=".repeat(60));
        info!("EMBEDDING COMPARISON OF COSINE DISTANCE");
        info!("{}", "=".repeat(60));

        info!("\nOne-Hot Embeddings:");
        self.print_distance_matrix(&self.one_hot_distances, "One-Hot");

        info!("\nLearned Embeddings:");
        self.print_distance_matrix(&self.learned_distances, "Learned");

        let oh_stats = self.compute_stats(&self.one_hot_distances);
        let learned_stats = self.compute_stats(&self.learned_distances);

        info!("\n{}", "-".repeat(60));
        info!("Distance Statistics:");
        info!("{}", "-".repeat(60));
        info!(
            "One-Hot - Mean: {:.4}, Std: {:.4}, Min: {:.4}, Max: {:.4}",
            oh_stats.0, oh_stats.1, oh_stats.2, oh_stats.3
        );
        info!(
            "Learned - Mean: {:.4}, Std: {:.4}, Min: {:.4}, Max: {:.4}",
            learned_stats.0, learned_stats.1, learned_stats.2, learned_stats.3
        );
        info!("{}", "=".repeat(60));
    }

    fn print_distance_matrix(&self, distances: &[Vec<f32>], label: &str) {
        info!("\n{} Distance Matrix:", label);
        let header: String = format!(
            "     {}",
            self.token_ids
                .iter()
                .map(|&id| format!("{:>6} ", format!("T{}", id)))
                .collect::<String>()
        );
        info!("{}", header);

        for (i, &id) in self.token_ids.iter().enumerate() {
            let row: String = distances[i]
                .iter()
                .map(|&d| format!("{:>6.3} ", d))
                .collect();
            info!("T{:<3} {}", id, row);
        }
    }

    fn compute_stats(&self, distances: &[Vec<f32>]) -> (f32, f32, f32, f32) {
        let mut all_distances = Vec::new();
        for i in 0..distances.len() {
            for j in (i + 1)..distances[i].len() {
                all_distances.push(distances[i][j]);
            }
        }

        if all_distances.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mean = all_distances.iter().sum::<f32>() / all_distances.len() as f32;
        let variance = all_distances
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / all_distances.len() as f32;
        let std = variance.sqrt();
        let min = all_distances.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = all_distances
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        (mean, std, min, max)
    }
}
