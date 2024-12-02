use bytes::Buf;
use flate2::bufread::GzDecoder;
use itertools::Itertools;
use jzon::parse;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Field,
};
use png::Decoder;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    fs::File,
    io::{Read, Write},
    path::Path,
};

use anyhow::{anyhow, bail, Result};

use crate::utils::DatasetPartition;

pub fn read_c4_realnewslike(c4_dir: &str, part: u64) -> Result<Vec<String>> {
    let path = format!("{c4_dir}/realnewslike/c4-train.{part:05}-of-00512.json.gz");
    let mut file = File::open(path)?;
    let mut buf: Vec<u8> = Vec::new();
    file.read_to_end(&mut buf)?;
    drop(file);

    let mut gz = GzDecoder::new(&buf[..]);
    let mut s = String::new();
    gz.read_to_string(&mut s)?;

    let mut result: Vec<String> = Vec::new();
    for line in s.splitn(s.len(), '\n') {
        let json = parse(line);
        if json.is_err() {
            break;
        }
        let json = json?;
        if json["text"].is_string() {
            result.push(
                json["text"]
                    .as_str()
                    .ok_or(anyhow!("parsing failure"))?
                    .to_owned(),
            );
        }
    }

    Ok(result)
}

pub fn read_tinystories(ts_dir: &str) -> Result<Vec<String>> {
    let path = format!("{ts_dir}/TinyStories-train.txt");
    let mut file = File::open(path)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    let mut result: Vec<String> = Vec::new();
    for line in s.splitn(s.len(), "\n<|endoftext|>\n") {
        result.push(line.to_owned());
    }

    Ok(result)
}

pub fn read_wikitext_test(wikitext_dir: &str) -> Result<Vec<String>> {
    let mut result: Vec<String> = Vec::new();
    let paths = vec![format!(
        "{wikitext_dir}/wikitext-2-v1/test-00000-of-00001.parquet"
    )];

    for path in paths {
        let path = Path::new(&path);
        let file = File::open(&path)?;
        let reader = SerializedFileReader::new(file).unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();

        for row in row_group_reader.get_row_iter(None)? {
            let row = row?;
            let field = &row.into_columns()[0].1;
            let row_str = if let Field::Str(s) = field {
                Some(s)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing Wikitext parquet file"))?;
            result.push(row_str.to_string());
        }
    }

    Ok(result)
}

pub fn read_wikitext(wikitext_dir: &str) -> Result<Vec<String>> {
    let mut result: Vec<String> = Vec::new();
    let paths = vec![
        format!("{wikitext_dir}/wikitext-2-v1/train-00000-of-00001.parquet"),
        // format!("{wikitext_dir}/wikitext-2-v1/train-00001-of-00002.parquet"),
    ];

    for path in paths {
        let path = Path::new(&path);
        let file = File::open(&path)?;
        let reader = SerializedFileReader::new(file).unwrap();
        let row_group_reader = reader.get_row_group(0).unwrap();

        for row in row_group_reader.get_row_iter(None)? {
            let row = row?;
            let field = &row.into_columns()[0].1;
            let row_str = if let Field::Str(s) = field {
                Some(s)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing Wikitext parquet file"))?;
            result.push(row_str.to_string());
        }
    }

    Ok(result)
}

// returns a tuple of the samples and classes
pub fn read_imdb(imdb_dir: &str, partition: DatasetPartition) -> Result<(Vec<String>, Vec<u8>)> {
    let mut result: Vec<String> = Vec::new();
    let mut classes: Vec<u8> = Vec::new();

    let path = match partition {
        DatasetPartition::Train => format!("{imdb_dir}/plain_text/train-00000-of-00001.parquet"),
        DatasetPartition::Test => format!("{imdb_dir}/plain_text/test-00000-of-00001.parquet"),
        DatasetPartition::Validation => bail!("Imbd does not have a validation set"),
    };

    let file = File::open(&path)?;
    let reader = SerializedFileReader::new(file).unwrap();
    for gp_idx in 0..reader.num_row_groups() {
        let row_group_reader = reader.get_row_group(gp_idx).unwrap();

        for row in row_group_reader.get_row_iter(None)? {
            let row = row?;

            let columns = row.into_columns();
            let row_str = if let Field::Str(s) = &columns[0].1 {
                Some(s)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing IMDB parquet file"))?;

            let class = if let Field::Long(c) = &columns[1].1 {
                Some(*c)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing IMDB parquet file"))?;
            result.push(row_str.to_string());
            classes.push(class as u8);
        }
    }

    Ok((result, classes))
}

fn read_mnist_like_dataset(path: String) -> Result<(Vec<Vec<u8>>, Vec<u8>)> {
    let mut result: Vec<Vec<u8>> = Vec::new();
    let mut classes: Vec<u8> = Vec::new();
    let path = Path::new(&path);
    let file = File::open(&path)?;
    let reader = SerializedFileReader::new(file).unwrap();
    for gp_idx in 0..reader.num_row_groups() {
        let row_group_reader = reader.get_row_group(gp_idx).unwrap();

        for outer_row in row_group_reader.get_row_iter(None)? {
            let outer_row = outer_row?.clone().into_columns();
            let bytes_field = &outer_row[0].1;
            let label_field = &outer_row[1].1;
            let class = if let Field::Long(x) = label_field {
                *x as u8
            } else {
                bail!("error parsing parquet file")
            };

            let row = if let Field::Group(r) = &bytes_field {
                Some(r)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing parquet file"))?
            .clone();
            let field = row.clone().into_columns()[0].clone().1;

            let png_bytes = if let Field::Bytes(b) = field {
                Some(b)
            } else {
                None
            }
            .ok_or(anyhow!("error parsing parquet file"))?
            .data()
            .to_vec();

            let tmp_path = format!(
                "{}/tmp.png",
                path.parent().unwrap_or(Path::new(".")).to_str().unwrap()
            );
            let tmp_path = Path::new(&tmp_path);
            let mut tmp = File::create(tmp_path)?;
            tmp.write_all(&png_bytes)?;
            drop(tmp);

            let decoder = Decoder::new(File::open(tmp_path)?);
            let mut reader = decoder.read_info().unwrap();
            // Allocate the output buffer.
            let mut buf = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf)?;
            let bytes = &buf[..info.buffer_size()];

            if bytes.len() != 28 * 28 {
                bail!(
                    "error parsing parquet file. Expected array with length {}, got {}",
                    28 * 28,
                    bytes.len()
                );
            }

            result.push(bytes.to_vec());
            classes.push(class);
        }
    }

    Ok((result, classes))
}

/// returns a tuple of the samples and classes
pub fn read_fashion_mnist(
    fashion_mnist_dir: &str,
    partition: DatasetPartition,
) -> Result<(Vec<Vec<u8>>, Vec<u8>)> {
    let path = match partition {
        DatasetPartition::Train => {
            format!("{fashion_mnist_dir}/fashion_mnist/train-00000-of-00001.parquet")
        }
        DatasetPartition::Test => {
            format!("{fashion_mnist_dir}/fashion_mnist/test-00000-of-00001.parquet")
        }
        DatasetPartition::Validation => bail!("FashionMnist has no validation set"),
    };
    read_mnist_like_dataset(path)
}
/// returns a tuple of the samples and classes
pub fn read_mnist(mnist_dir: &str, partition: DatasetPartition) -> Result<(Vec<Vec<u8>>, Vec<u8>)> {
    let path = match partition {
        DatasetPartition::Train => format!("{mnist_dir}/mnist/train-00000-of-00001.parquet"),
        DatasetPartition::Test => format!("{mnist_dir}/mnist/test-00000-of-00001.parquet"),
        DatasetPartition::Validation => bail!("Mnist has no validation set"),
    };
    read_mnist_like_dataset(path)
}

/// returns a tuple of the samples and classes, where ham is 0 and spam is 1
pub fn read_spam(spam_dir: &str, partition: DatasetPartition) -> Result<(Vec<String>, Vec<u8>)> {
    const SEED: u64 = 42;
    const TRAIN_PROB: f64 = 0.5;
    let train_partition = partition == DatasetPartition::Train;
    let path = format!("{spam_dir}/enron_spam_data.csv");
    let mut reader = csv::Reader::from_path(path)?;

    let mut emails = Vec::new();
    let mut classes = Vec::new();

    let mut rng = StdRng::seed_from_u64(SEED);
    for result in reader.records() {
        let train = rng.gen_bool(TRAIN_PROB);

        if train != train_partition {
            continue;
        }
        let rec = result?;

        let mut email = rec[1].to_string();
        email.push_str(&rec[2]);
        emails.push(email);

        let class = match &rec[3] {
            "spam" => 1,
            "ham" => 0,
            _ => bail!("unexpected class"),
        };
        classes.push(class as u8);
    }
    Ok((emails, classes))
}

pub fn read_file_to_string(path: &str) -> Result<String> {
    let path = Path::new(&path);
    let mut file = File::open(&path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;

    Ok(buf)
}

pub fn read_cifar10(
    cifar10_dir: &str,
    partition: DatasetPartition,
) -> Result<(Vec<Vec<u8>>, Vec<u8>)> {
    let paths = match partition {
        DatasetPartition::Train => (1..=5)
            .map(|i| format!("{cifar10_dir}/cifar-10-batches-bin/data_batch_{i}.bin"))
            .collect_vec(),
        DatasetPartition::Test => {
            vec![format!("{cifar10_dir}/cifar-10-batches-bin/test_batch.bin")]
        }
        DatasetPartition::Validation => bail!("Cifar10 has no validation set"),
    };

    let mut vecs = Vec::new();
    let mut classes = Vec::new();
    for path in paths {
        let mut buf = Vec::new();
        File::open(path)?.read_to_end(&mut buf)?;
        let buf_ptr = &mut buf.as_slice();

        while buf_ptr.len() > 0 {
            classes.push(buf_ptr.get_u8());

            // Make a grayscale image via averaging
            let mut sum_vec = vec![0; 1024];
            for _rgb_idx in 0..3 {
                for i in 0..1024 {
                    sum_vec[i] = sum_vec[i].max(buf_ptr.get_u8());
                }
            }
            // let mut vec = Vec::new();
            // for _rgb_idx in 0..3 {
            //     for _i in 0..1024 {
            //         vec.push(buf_ptr.get_u8());
            //     }
            // }
            // vecs.push(vec);

            vecs.push(sum_vec.into_iter().map(|x| (x) as u8).collect_vec());
        }
    }

    Ok((vecs, classes))
}

pub fn read_fasta(path: &str) -> Result<Vec<String>> {
    let path = Path::new(&path);
    let mut file = File::open(&path)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;

    let mut result: Vec<String> = Vec::new();
    let mut curr_string = String::new();
    for line in s.splitn(s.len(), '\n') {
        if line.len() == 0 {
            continue;
        }
        if line.chars().next().unwrap() == '>' {
            if curr_string.len() > 0 {
                result.push(curr_string);
                curr_string = String::new();
            }
            continue;
        }
        curr_string.push_str(line);
    }

    if curr_string.len() > 0 {
        result.push(curr_string);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{read_fasta, read_spam};

    #[test]
    fn dna() {
        let res =
            read_fasta("/home/nsagan/LZ78-implementation/lz78_rust/data/genes/ndufs2.fna").unwrap();
        println!("{res:?} {}", res.len())
    }

    #[test]
    fn spam() {
        read_spam(
            "/home/nsagan/LZ78-implementation/lz78_rust/data/enron_spam_data",
            super::DatasetPartition::Train,
        )
        .unwrap();
    }
}
