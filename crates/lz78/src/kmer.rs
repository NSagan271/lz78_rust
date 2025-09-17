use hashbrown::{HashMap, HashSet};
use std::io::{BufRead, BufReader};

fn base4_char_to_int(c: char) -> u64 {
    match c {
        'A' => 0,
        'C' => 1,
        'G' => 2,
        'T' => 3,
        _ => 0,
    }
}

fn base4_char_to_rev_int(c: char) -> u64 {
    match c {
        'A' => 3,
        'C' => 2,
        'G' => 1,
        'T' => 0,
        _ => 3,
    }
}

fn kmer_to_int(kmer: &str, canonical: bool) -> (u64, u64, u64) {
    // If canonical, return the lexicographically smaller of kmer and its reverse complement
    let revcomp = kmer
        .chars()
        .rev()
        .map(|c| match c {
            'A' => 'T',
            'C' => 'G',
            'G' => 'C',
            'T' => 'A',
            _ => 'N',
        })
        .collect::<String>();

    let mut result = 0;
    let mut base: u64 = 1;
    for c in kmer.chars().rev() {
        result |= base * base4_char_to_int(c);
        base <<= 2;
    }

    let mut revcomp_result = 0;
    let mut base: u64 = 1;
    for c in revcomp.chars().rev() {
        revcomp_result |= base * base4_char_to_int(c);
        base <<= 2;
    }

    (
        result,
        revcomp_result,
        if canonical {
            result.min(revcomp_result)
        } else {
            result
        },
    )
}

pub fn counts_from_file(path: &str, k: usize, canonical: bool) -> HashMap<u64, u64> {
    let mut counts = HashMap::new();
    let file = std::fs::File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        if let Ok(line) = line {
            let parts: Vec<&str> = line.trim().split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            let km = parts[0];
            let ct = parts[parts.len() - 1];
            if km.len() != k {
                continue;
            }
            if let Ok(count) = ct.parse::<u64>() {
                let kmer_int = kmer_to_int(km, canonical).2;
                *counts.entry(kmer_int).or_insert(0) += count;
            }
        }
    }
    counts
}

pub fn get_background_probs(path: &str, k: usize, canonical: bool) -> Option<HashMap<u64, f64>> {
    let bg_counts = counts_from_file(path, k, canonical);
    let tot: u64 = bg_counts.values().sum();
    if tot > 0 {
        Some(
            bg_counts
                .iter()
                .map(|(&km, &ct)| (km, ct as f64 / tot as f64))
                .collect(),
        )
    } else {
        None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum FeatureMode {
    Count,
    Binary,
}

#[derive(Clone)]
pub struct KmerMultinomial {
    k: usize,
    alpha: f64,
    feature_mode: FeatureMode,
    canonical: bool,
    background_priors: Option<HashMap<u64, f64>>,
    class_probs: Vec<HashMap<u64, f64>>,
}

impl KmerMultinomial {
    pub fn new(
        k: usize,
        alpha: f64,
        feature_mode: FeatureMode,
        canonical: bool,
        background_priors: Option<HashMap<u64, f64>>,
    ) -> Self {
        KmerMultinomial {
            k,
            alpha,
            feature_mode,
            canonical,
            background_priors,
            class_probs: Vec::new(),
        }
    }

    pub fn fit_from_counts(&mut self, class_counts: &Vec<HashMap<u64, u64>>) {
        let mut vocab = HashSet::new();
        for counts in class_counts {
            for &kmer in counts.keys() {
                vocab.insert(kmer);
            }
        }
        if let Some(bg) = &self.background_priors {
            for &kmer in bg.keys() {
                vocab.insert(kmer);
            }
        }
        let v = vocab.len().max(1) as f64;

        // Normalize background priors over current vocab; treat as unit-mass prior (no user β)
        let bg_normalized = if let Some(bg) = &self.background_priors {
            let tot_bg: f64 = vocab
                .iter()
                .map(|&km| bg.get(&km).cloned().unwrap_or(0.0))
                .sum();
            if tot_bg > 0.0 {
                let mut bg_norm = HashMap::new();
                for &km in vocab.iter() {
                    bg_norm.insert(km, bg.get(&km).cloned().unwrap_or(0.0) / tot_bg);
                }
                Some(bg_norm)
            } else {
                None
            }
        } else {
            None
        };

        self.class_probs.clear();
        for counts_c in class_counts {
            let total_count: u64 = counts_c.values().sum();
            let denom = total_count as f64
                + (self.alpha * v)
                + if bg_normalized.is_some() { 1.0 } else { 0.0 };
            let mut probs = HashMap::new();
            for &km in vocab.iter() {
                let prior = if let Some(bg) = &bg_normalized {
                    bg.get(&km).cloned().unwrap_or(0.0)
                } else {
                    0.0
                };
                let count = counts_c.get(&km).cloned().unwrap_or(0);
                probs.insert(km, (count as f64 + self.alpha + prior) / denom);
            }
            self.class_probs.push(probs);
        }
    }

    pub fn fit_from_files(&mut self, path_per_class: Vec<&str>) {
        let mut counts_per_class = Vec::new();
        for path in path_per_class {
            let counts = counts_from_file(path, self.k, self.canonical);
            counts_per_class.push(counts);
        }
        self.fit_from_counts(&counts_per_class);
    }

    fn seq_counts(&self, seq: &str) -> HashMap<u64, u64> {
        let l = seq.len();
        if l < self.k {
            return HashMap::new();
        }

        let mut out = HashMap::new();

        let (mut kmer, mut rev_kmer, mut canon_kmer) = kmer_to_int(&seq[0..self.k], self.canonical);
        out.insert(canon_kmer, 1);

        let mask = (1u64 << (2 * self.k)) - 1; // mask to keep only last k-mer bits
        let bytes = seq.as_bytes();

        // Rolling k-mer updates
        for i in self.k..l {
            kmer = ((kmer << 2) & mask) | base4_char_to_int(bytes[i] as char);
            rev_kmer =
                (rev_kmer >> 2) | (base4_char_to_rev_int(bytes[i] as char) << (2 * (self.k - 1)));
            canon_kmer = if self.canonical {
                kmer.min(rev_kmer)
            } else {
                kmer
            };

            if self.feature_mode == FeatureMode::Binary {
                out.insert(canon_kmer, 1); // presence only
            } else {
                *out.entry(canon_kmer).or_insert(0) += 1;
            }
        }

        out
    }

    pub fn predict_one(&self, seq: &str) -> u64 {
        let x = self.seq_counts(seq);
        let mut best_c: Option<usize> = None;
        let mut best_s = -1e300;
        for (c, p) in self.class_probs.iter().enumerate() {
            let mut s = 0.0;
            for (&km, &cnt) in x.iter() {
                let pt = p.get(&km).cloned().unwrap_or(1e-12);
                s += (cnt as f64) * pt.ln();
            }
            if s > best_s {
                best_s = s;
                best_c = Some(c);
            }
        }
        best_c.unwrap_or(0) as u64
    }

    pub fn predict_parallel(&self, seqs: Vec<&str>, num_threads: usize) -> Vec<u64> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        pool.install(|| seqs.iter().map(|&s| self.predict_one(s)).collect())
    }
}
