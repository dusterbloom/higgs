// SPDX-License-Identifier: Apache-2.0
//! Token importance scoring for SpecPrefill.

use mlx_rs::{Array, Exception};

/// Token importance scoring result.
#[derive(Debug, Clone)]
pub struct TokenImportance {
    pub scores: Array,
    pub n_tokens: usize,
}

impl TokenImportance {
    pub fn select_chunks(&self, keep_pct: f32, chunk_size: usize) -> Result<Vec<usize>, Exception> {
        let n = self.n_tokens;
        if keep_pct >= 1.0 {
            return Ok((0..n).collect());
        }
        let scores: Vec<f32> = self.scores.try_into()?;
        let n_chunks = (n + chunk_size - 1) / chunk_size;
        let keep_n = std::cmp::max(1, (n_chunks as f32 * keep_pct).ceil() as usize);
        let mut chunk_scores: Vec<(usize, f32)> = Vec::with_capacity(n_chunks);
        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, n);
            let avg = scores[start..end].iter().sum::<f32>() / (end - start) as f32;
            chunk_scores.push((i, avg));
        }
        chunk_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept: Vec<usize> = chunk_scores.into_iter().take(keep_n).map(|(i, _)| i).collect();
        kept.sort();
        let mut indices = Vec::new();
        for ci in kept {
            let start = ci * chunk_size;
            let end = std::cmp::min(start + chunk_size, n);
            indices.extend(start..end);
        }
        Ok(indices)
    }

    pub fn select_topk(&self, k: usize) -> Result<Vec<usize>, Exception> {
        let scores: Vec<f32> = self.scores.try_into()?;
        let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut indices: Vec<usize> = indexed.into_iter().take(k).map(|(i, _)| i).collect();
        indices.sort();
        Ok(indices)
    }
}

pub fn score_tokens_uniform(n_tokens: usize) -> Result<TokenImportance, Exception> {
    let uniform = 1.0 / n_tokens as f32;
    let scores = Array::from_full(&[n_tokens as i32], uniform)?;
    Ok(TokenImportance { scores, n_tokens })
}

pub fn compute_keep_rate(prompt_len: usize) -> f32 {
    if prompt_len < 8192 { 1.0 }
    else if prompt_len < 16384 { 0.30 }
    else if prompt_len < 32768 { 0.25 }
    else { 0.20 }
}
