use crate::bpe::trainer::TrainingMetrics;
use log::info;
use plotters::prelude::*;
use std::error::Error;

/// Plot BPE training metrics: vocabulary growth and token compression
///
/// Educational Purpose: Visualize how BPE learns and compresses text
/// Theory: BPE progressively learns subword units, growing vocabulary while
/// reducing token count (compression).
///
/// Visualization shows:
/// 1. Vocabulary size growth over merge iterations
/// 2. Token count reduction (compression effect)
/// 3. Trade-off between vocab size and sequence length
pub fn plot_training_metrics(
    metrics: &TrainingMetrics,
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.titled(
        "BPE Training Metrics: Vocabulary Growth & Token Compression",
        ("sans-serif", 30),
    )?;

    // Split into 3 areas: vocab growth, token reduction, compression ratio
    let areas = root.split_evenly((3, 1));

    // Plot 1: Vocabulary Size Growth
    plot_vocab_growth(&areas[0], metrics)?;

    // Plot 2: Token Count Reduction (Compression)
    plot_token_compression(&areas[1], metrics)?;

    // Plot 3: Merge Frequency Distribution
    plot_merge_frequencies(&areas[2], metrics)?;

    root.present()?;
    info!("Training metrics plot saved to: {}", output_path);
    Ok(())
}

/// Plot vocabulary size as it grows with each merge
///
/// Theory: Each merge adds one new token (the merged pair) to vocabulary.
/// Vocabulary grows linearly with number of merges (in addition to base chars).
fn plot_vocab_growth(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    metrics: &TrainingMetrics,
) -> Result<(), Box<dyn Error>> {
    let vocab_sizes = &metrics.vocab_sizes;
    if vocab_sizes.is_empty() {
        return Ok(());
    }

    let max_vocab = *vocab_sizes.iter().max().unwrap();
    let min_vocab = *vocab_sizes.iter().min().unwrap();

    let mut chart = ChartBuilder::on(area)
        .caption("Vocabulary Size Growth", ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0usize..vocab_sizes.len(),
            min_vocab.saturating_sub(5)..max_vocab + 5,
        )?;

    chart
        .configure_mesh()
        .x_desc("Merge Iteration")
        .y_desc("Vocabulary Size")
        .draw()?;

    // Draw line series
    chart.draw_series(LineSeries::new(
        vocab_sizes.iter().enumerate().map(|(i, &v)| (i, v)),
        &BLUE,
    ))?;

    // Add circle markers
    chart.draw_series(
        vocab_sizes
            .iter()
            .enumerate()
            .map(|(i, &v)| Circle::new((i, v), 3, BLUE.filled())),
    )?;

    // Add annotation for initial and final vocab size
    let initial = vocab_sizes[0];
    let final_size = *vocab_sizes.last().unwrap();

    chart.draw_series(std::iter::once(Text::new(
        format!("Initial: {}", initial),
        (0, initial + 2),
        ("sans-serif", 12).into_font().color(&BLACK),
    )))?;

    chart.draw_series(std::iter::once(Text::new(
        format!("Final: {}", final_size),
        (vocab_sizes.len() - 1, final_size + 2),
        ("sans-serif", 12).into_font().color(&BLACK),
    )))?;

    Ok(())
}

/// Plot token count reduction showing compression effect
///
/// Theory: As we merge frequent pairs, total token count decreases.
/// This shows BPE's compression capability - representing same text
/// with fewer tokens using learned subword units.
fn plot_token_compression(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    metrics: &TrainingMetrics,
) -> Result<(), Box<dyn Error>> {
    let token_counts = &metrics.token_counts;
    if token_counts.is_empty() {
        return Ok(());
    }

    let max_count = *token_counts.iter().max().unwrap();
    let min_count = *token_counts.iter().min().unwrap();

    let mut chart = ChartBuilder::on(area)
        .caption("Token Count Reduction (Compression)", ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0usize..token_counts.len(),
            min_count.saturating_sub(10)..max_count + 10,
        )?;

    chart
        .configure_mesh()
        .x_desc("Merge Iteration")
        .y_desc("Total Token Count")
        .draw()?;

    // Draw line series showing reduction
    chart.draw_series(LineSeries::new(
        token_counts.iter().enumerate().map(|(i, &c)| (i, c)),
        &RED,
    ))?;

    // Add filled area to show compression visually
    chart.draw_series(AreaSeries::new(
        token_counts.iter().enumerate().map(|(i, &c)| (i, c)),
        min_count.saturating_sub(10),
        &RED.mix(0.2),
    ))?;

    // Add markers
    chart.draw_series(
        token_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| Circle::new((i, c), 3, RED.filled())),
    )?;

    // Calculate and display compression stats
    let initial = token_counts[0];
    let final_count = *token_counts.last().unwrap();
    let compression_ratio = metrics.compression_ratio();
    let reduction = initial - final_count;
    let reduction_pct = (reduction as f32 / initial as f32) * 100.0;

    chart.draw_series(std::iter::once(Text::new(
        format!("Initial: {} tokens", initial),
        (0, initial + 5),
        ("sans-serif", 12).into_font().color(&BLACK),
    )))?;

    chart.draw_series(std::iter::once(Text::new(
        format!("Final: {} tokens", final_count),
        (token_counts.len() - 1, final_count + 5),
        ("sans-serif", 12).into_font().color(&BLACK),
    )))?;

    // Add compression stats in the middle
    let mid_x = token_counts.len() / 2;
    let mid_y = (max_count + min_count) / 2;

    chart.draw_series(std::iter::once(Text::new(
        format!(
            "Compression: {:.2}x | Reduced: -{:.1}%",
            compression_ratio, reduction_pct
        ),
        (mid_x, mid_y),
        ("sans-serif", 14).into_font().color(&RED),
    )))?;

    Ok(())
}

/// Plot merge frequencies to understand what pairs were most important
///
/// Theory: Early merges have higher frequencies (most common pairs).
/// This shows the frequency distribution and helps understand
/// what patterns BPE learned from the corpus.
fn plot_merge_frequencies(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    metrics: &TrainingMetrics,
) -> Result<(), Box<dyn Error>> {
    let merge_history = &metrics.merge_history;
    if merge_history.is_empty() {
        return Ok(());
    }

    let frequencies: Vec<usize> = merge_history.iter().map(|(_, freq)| *freq).collect();
    let max_freq = *frequencies.iter().max().unwrap();

    let mut chart = ChartBuilder::on(area)
        .caption("Merge Frequencies (Most Common Pairs)", ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0usize..frequencies.len(), 0usize..max_freq + 5)?;

    chart
        .configure_mesh()
        .x_desc("Merge Order (Earlier = More Frequent)")
        .y_desc("Pair Frequency")
        .draw()?;

    // Draw bars for frequencies
    chart.draw_series(
        frequencies
            .iter()
            .enumerate()
            .map(|(i, &freq)| Rectangle::new([(i, 0), (i + 1, freq)], GREEN.filled())),
    )?;

    // Add outline to bars
    chart.draw_series(
        frequencies
            .iter()
            .enumerate()
            .map(|(i, &freq)| Rectangle::new([(i, 0), (i + 1, freq)], GREEN.stroke_width(1))),
    )?;

    // Label top 3 merges
    let top_n = 3.min(merge_history.len());
    for i in 0..top_n {
        let ((pair_a, pair_b), freq) = &merge_history[i];
        let label = format!("\"{}{}\":{}", pair_a, pair_b, freq);
        chart.draw_series(std::iter::once(Text::new(
            label,
            (i, *freq + 2),
            ("sans-serif", 10).into_font().color(&BLACK),
        )))?;
    }

    Ok(())
}

pub fn plot_distance_comparison(
    one_hot_distances: &[Vec<f32>],
    learned_distances: &[Vec<f32>],
    token_ids: &[usize],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.titled(
        "One-Hot vs Learned Embeddings: Cosine Distance Comparison",
        ("sans-serif", 30),
    )?;

    let areas = root.split_evenly((2, 1));
    let upper = &areas[0];
    let lower = &areas[1];

    plot_heatmap(
        upper,
        one_hot_distances,
        token_ids,
        "One-Hot Embeddings",
        0.0,
        2.0,
    )?;

    plot_heatmap(
        lower,
        learned_distances,
        token_ids,
        "Learned Embeddings",
        0.0,
        2.0,
    )?;

    root.present()?;
    info!("\nPlot saved to: {}", output_path);
    Ok(())
}

fn plot_heatmap(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    distances: &[Vec<f32>],
    token_ids: &[usize],
    title: &str,
    min_val: f32,
    max_val: f32,
) -> Result<(), Box<dyn Error>> {
    let n = distances.len();

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0..n, 0..n)?;

    chart
        .configure_mesh()
        .x_desc("Token ID")
        .y_desc("Token ID")
        .x_label_formatter(&|x| {
            if *x < token_ids.len() {
                format!("{}", token_ids[*x])
            } else {
                String::new()
            }
        })
        .y_label_formatter(&|y| {
            if *y < token_ids.len() {
                format!("{}", token_ids[*y])
            } else {
                String::new()
            }
        })
        .draw()?;

    for i in 0..n {
        for j in 0..n {
            let distance = distances[i][j];
            let normalized = ((distance - min_val) / (max_val - min_val)).clamp(0.0, 1.0);

            let color = if i == j {
                RGBColor(200, 200, 200)
            } else {
                let red = (normalized * 255.0) as u8;
                let blue = ((1.0 - normalized) * 255.0) as u8;
                RGBColor(red, 0, blue)
            };

            chart.draw_series(std::iter::once(Rectangle::new(
                [(j, i), (j + 1, i + 1)],
                color.filled(),
            )))?;
        }
    }

    Ok(())
}

pub fn plot_distance_histograms(
    one_hot_distances: &[Vec<f32>],
    learned_distances: &[Vec<f32>],
    output_path: &str,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let root = root.titled("Distance Distribution Comparison", ("sans-serif", 30))?;

    let mut oh_dists = Vec::new();
    let mut learned_dists = Vec::new();

    for i in 0..one_hot_distances.len() {
        for j in (i + 1)..one_hot_distances[i].len() {
            oh_dists.push(one_hot_distances[i][j]);
            learned_dists.push(learned_distances[i][j]);
        }
    }

    let areas = root.split_evenly((1, 2));
    let left = &areas[0];
    let right = &areas[1];

    plot_histogram(left, &oh_dists, "One-Hot", &BLUE)?;

    plot_histogram(right, &learned_dists, "Learned", &RED)?;

    root.present()?;
    info!("Histogram plot saved to: {}", output_path);
    Ok(())
}

fn plot_histogram(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    data: &[f32],
    title: &str,
    color: &RGBColor,
) -> Result<(), Box<dyn Error>> {
    let num_bins = 20;
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let bin_width = (max - min) / num_bins as f32;

    let mut bins = vec![0; num_bins];
    for &val in data {
        let bin_idx = ((val - min) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(area)
        .caption(title, ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min..max, 0..max_count)?;

    chart
        .configure_mesh()
        .x_desc("Distance")
        .y_desc("Count")
        .draw()?;

    chart.draw_series(bins.iter().enumerate().map(|(i, &count)| {
        let x0 = min + i as f32 * bin_width;
        let x1 = x0 + bin_width;
        Rectangle::new([(x0, 0), (x1, count)], color.filled())
    }))?;

    Ok(())
}
