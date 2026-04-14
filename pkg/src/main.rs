/* Create and use suffix arrays for deduplicating language model datasets.
 *
 * A suffix array A for a sequence S is a datastructure that contains all
 * suffixes of S in sorted order. To be space efficient, instead of storing
 * the actual suffix, we just store the pointer to the start of the suffix.
 * To be time efficient, it uses fancy algorithms to not require quadratic
 * (or worse) work. If we didn't care about either, then we could literally
 * just define (in python)
 * A = sorted(S[i:] for i in range(len(S)))
 *
 * Suffix arrays are amazing because they allow us to run lots of string
 * queries really quickly, while also only requiring an extra 8N bytes of
 * storage (one 64-bit pointer for each byte in the sequence).
 *
 * This code is designed to work with Big Data (TM) and most of the
 * complexity revolves around the fact that we do not require the
 * entire suffix array to fit in memory. In order to keep things managable,
 * we *do* require that the original string fits in memory. However, even
 * the largest language model datasets (e.g., C4) are a few hundred GB
 * which on todays machines does fit in memory.
 *
 * With all that amazing stuff out of the way, just a word of warning: this
 * is the first program I've ever written in rust. I still don't actually
 * understand what borrowing something means, but have found that if I
 * add enough &(&&x.copy()).clone() then usually the compiler just loses
 * all hope in humanity and lets me do what I want. I apologize in advance
 * to anyone actually does know rust and wants to lock me in a small room
 * with the Rust Book by Klabnik & Nichols until I repent for my sins.
 * (It's me, two months in the future. I now more or less understand how
 * to borrow. So now instead of the code just being all awful, you'll get
 * a nice mix of sane rust and then suddenly OH-NO-WHAT-HAVE-YOU-DONE-WHY!?!)
 */

use std::time::Instant;
use std::fs;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::BufReader;
use std::io::SeekFrom;
use std::fs::File;
use std::io::prelude::*;

extern crate filebuffer;
extern crate zstd;
extern crate crossbeam;
extern crate clap;
extern crate glob;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use clap::{Parser, Subcommand};
use glob::glob;

mod table_u8;
mod table_u16;
mod table_u32;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {

    #[clap(arg_required_else_help = true)]
    MakePart {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        parts_dir: String,
        #[clap(short, long)]
        start_byte: usize,
        #[clap(short, long)]
        end_byte: usize,
        #[clap(short, long)]
        ratio: usize,
        #[clap(short, long)]
        token_width: usize,
    },

    Merge {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        parts_dir: String,
        #[clap(short, long)]
        merged_dir: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
        #[clap(long, default_value_t = 100000)]
        hacksize: usize,
        #[clap(short, long)]
        ratio: usize,
        #[clap(short, long)]
        token_width: usize,
    },

    Concat {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        merged_dir: String,
        #[clap(short, long)]
        merged_file: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
        #[clap(short, long)]
        ratio: usize,
        #[clap(short, long)]
        token_width: usize,
    },

    BuildBloom {
        #[clap(short, long)]
        data_file: String,
        #[clap(long)]
        bloom_file: String,
        #[clap(long)]
        num_bits: u64,
        #[clap(long, default_value_t = 7)]
        num_hashes: u32,
        #[clap(long, default_value_t = 4)]
        max_ngram_n: u32,
        #[clap(short, long)]
        token_width: usize,
    },

    BuildBigram {
        #[clap(short, long)]
        data_file: String,
        #[clap(long)]
        table_file: String,
        #[clap(long)]
        bigram_file: String,
        #[clap(short, long)]
        token_width: usize,
        #[clap(short, long)]
        ratio: usize,
    }

}

/* Convert a uint64 array to a uint8 array.
 * This doubles the memory requirements of the program, but in practice
 * we only call this on datastructures that are smaller than our assumed
 * machine memory so it works.
 */
pub fn to_bytes(input: &[u64], size_width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_width * input.len());

    for value in input {
        bytes.extend(&value.to_le_bytes()[..size_width]);
    }
    bytes
}

/* Convert a uint8 array to a uint64. Only called on (relatively) small files. */
pub fn from_bytes(input: Vec<u8>, size_width: usize) -> Vec<u64> {
    assert!(input.len() % size_width == 0);
    let mut bytes:Vec<u64> = Vec::with_capacity(input.len()/size_width);

    let mut tmp = [0u8; 8];
    // todo learn rust macros, hope they're half as good as lisp marcos
    // and if they are then come back and optimize this
    for i in 0..input.len()/size_width {
        tmp[..size_width].copy_from_slice(&input[i*size_width..i*size_width+size_width]);
        bytes.push(u64::from_le_bytes(tmp));
    }

    bytes
}

/* For a suffix array, just compute A[i], but load off disk because A is biiiiiiigggggg. */
fn table_load_disk(table:&mut BufReader<File>,
                   index: usize,
                   size_width: usize) -> usize{
    table.seek(std::io::SeekFrom::Start ((index*size_width) as u64)).expect ("Seek failed!");
    let mut tmp = [0u8; 8];
    table.read_exact(&mut tmp[..size_width]).unwrap();
    return u64::from_le_bytes(tmp) as usize;
}

/* Binary search to find where query happens to exist in text */
fn off_disk_position(text: &[u8], table: &mut BufReader<File>,
                     query: &[u8], size_width: usize, token_width: usize) -> usize {
    let (mut left, mut right) = (0, text.len() / token_width);
    while left < right {
        let mid = (left + right) / 2;
        if query < &text[table_load_disk(table, mid, size_width)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

/*
 * We're going to work with suffix arrays that are on disk, and we often want
 * to stream them top-to-bottom. This is a datastructure that helps us do that:
 * we read 1MB chunks of data at a time into the cache, and then fetch new data
 * when we reach the end.
 */
struct TableStream {
    file: BufReader<File>,
    cache: [u8; 8],
    size_width: usize
}

/* Make a table from a file path and a given offset into the table */
fn make_table(path: std::string::String,
              offset: usize,
              size_width: usize) -> TableStream {
    let mut table = TableStream {
        file: std::io::BufReader::with_capacity(1024*1024, fs::File::open(path).unwrap()),
        cache: [0u8; 8],
        size_width: size_width
    };
    table.file.seek (std::io::SeekFrom::Start ((offset*size_width) as u64)).expect ("Seek failed!");
    return table;
}

/* Get the next word from the suffix table. */
fn get_next_pointer_from_table_canfail(tablestream:&mut TableStream) -> u64 {
    let ok = tablestream.file.read_exact(&mut tablestream.cache[..tablestream.size_width]);
    let bad = match ok {
        Ok(_) => false,
        Err(_) => true,
    };
    if bad {
        return std::u64::MAX;
    }
    let out = u64::from_le_bytes(tablestream.cache);
    return out;
}

/*
 * Create a suffix array for a subsequence of bytes.
 * As with save, this method is linear in the number of bytes that are
 * being saved but the constant is rather high. This method does exactly
 * the same thing as save except on a range of bytes.
 */
fn cmd_make_part(fpath: &String, parts_dir: &String, start: u64, end: u64, ratio: usize, token_width: usize)   -> std::io::Result<()> {
    let now = Instant::now();
    println!("Opening up the dataset files");

    let space_available = std::fs::metadata(fpath.clone()).unwrap().len() as u64;
    assert!(start < end);
    assert!(end <= space_available);

    let mut text_ = vec![0u8; (end-start) as usize];
    let mut file = fs::File::open(fpath.clone()).unwrap();
    println!("Loading part of file from byte {} to {}", start, end);
    file.seek(std::io::SeekFrom::Start(start)).expect ("Seek failed!");
    file.read_exact(&mut text_).unwrap();

    assert!(text_.len() % token_width == 0);
    let table2:Vec<u64>;
    if token_width == 1 {
        let text = &text_;
        let st = table_u8::SuffixTable::new(text);
        let parts = st.into_parts();
        let table = parts.1;
        table2 = table.iter().map(|x| *x as u64).collect::<Vec<u64>>();
    } else if token_width == 2 {
        // LJC: Cast the buffer to tokens so we can build the SA for valid positions only
        // LJC: We have to use big-endian here to interpret into u16, because the suffix array operates on bytes
        let u16_text_: Vec<u16> = text_
            .chunks(2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]))
            .collect();
        let u16_text = &u16_text_;
        let st = table_u16::SuffixTable::new(u16_text);
        let parts = st.into_parts();
        let table = parts.1;
        // LJC: multiply every element of the table by 2, because the offsets are counted in bytes
        table2 = table.iter().map(|x| x*2).collect::<Vec<u64>>();
    } else if token_width == 4 {
        let u32_text_: Vec<u32> = text_
            .chunks(4)
            .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let u32_text = &u32_text_;
        let st = table_u32::SuffixTable::new(u32_text);
        let parts = st.into_parts();
        let table = parts.1;
        table2 = table.iter().map(|x| x*4).collect::<Vec<u64>>();
    } else {
        panic!("Unsupported token width: {}", token_width);
    }

    let mut buffer = File::create(format!("{}/{}-{}", parts_dir, start, end))?;
    let bufout = to_bytes(&table2, ratio);
    println!("Writing the suffix array at time t={}ms", now.elapsed().as_millis());
    buffer.write_all(&bufout)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

/*
 * A little bit of state for the merge operation below.
 * - suffix is suffix of one of the parts of the dataset we're merging;
this is the value we're sorting on
 * - position is the location of this suffix (so suffix = array[position..])
 * - table_index says which suffix array this suffix is a part of
 */
#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeState<'a> {
    suffix: &'a [u8],
    position: u64,
    table_index: usize,
    hacksize: usize,
}

impl<'a> Ord for MergeState<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix[..other.suffix.len().min(other.hacksize)].cmp(&self.suffix[..self.suffix.len().min(self.hacksize)])
    }
}

impl<'a> PartialOrd for MergeState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/*
 * Merge together M different suffix arrays (probably created with make-part).
 * That is, given strings S_i and suffix arrays A_i compute the suffix array
 * A* = make-suffix-array(concat S_i)
 * In order to do this we just implement mergesort's Merge operation on each
 * of the arrays A_i to construct a sorted array A*.
 *
 * This algorithm is *NOT A LINEAR TIME ALGORITHM* in the worst case. If you run
 * it on a dataset consisting entirely of the character A it will be quadratic.
 * Fortunately for us, language model datasets typically don't just repeat the same
 * character a hundred million times in a row. So in practice, it's linear time.
 *
 * There are thre complications here.
 *
 * As with selfsimilar_parallel, we can't fit all A_i into memory at once, and
 * we want to make things fast and so parallelize our execution. So we do the
 * same tricks as before to make things work.
 *
 * However we have one more problem. In order to know how to merge the final
 * few bytes of array S_0 into their correct, we need to know what bytes come next.
 * So in practice we make sure that S_{i}[-HACKSIZE:] === S_{i+1}[:HACKSIZE].
 * As long as HACKSIZE is longer than the longest potential match, everything
 * will work out correctly. (I did call it hacksize after all.....)
 * In practice this works. It may not for your use case if there are long duplicates.
 */
fn cmd_merge(fpath: &String, parts_dir: &String, merged_dir: &String, num_threads: i64, hacksize: usize, ratio: usize, token_width: usize)  -> std::io::Result<()> {

    let mut part_files:Vec<String> = glob(&format!("{}/*", parts_dir)).unwrap().map(|x| x.unwrap()).map(|x| x.to_str().unwrap().to_string()).collect();
    part_files.sort_by(|a, b| {
        let parts_a:Vec<&str> = a.split("/").last().unwrap().split("-").collect();
        let parts_b:Vec<&str> = b.split("/").last().unwrap().split("-").collect();
        assert!(parts_a.len() == 2);
        assert!(parts_b.len() == 2);
        let start_a = parts_a[0].parse::<u64>().unwrap();
        let start_b = parts_b[0].parse::<u64>().unwrap();
        start_a.cmp(&start_b)
    });
    let part_ranges:Vec<(u64, u64)> = part_files.iter().map(|x| {
        let parts:Vec<&str> = x.split("/").last().unwrap().split("-").collect();
        assert!(parts.len() == 2);
        let start = parts[0].parse::<u64>().unwrap();
        let end = parts[1].parse::<u64>().unwrap();
        (start, end)
    }).collect();

    let nn:usize = part_files.len();

    fn load_text2<'s,'t>(fpath: String, start: u64, end: u64) -> Vec<u8> {
        let space_available = std::fs::metadata(fpath.clone()).unwrap().len() as u64;
        assert!(start < end);
        assert!(end <= space_available);
        let mut text_ = vec![0u8; (end-start) as usize];
        let mut file = fs::File::open(fpath.clone()).unwrap();
        file.seek(std::io::SeekFrom::Start(start)).expect ("Seek failed!");
        file.read_exact(&mut text_).unwrap();
        return text_;
    }

    let texts:Vec<Vec<u8>> = (0..nn).map(|x| load_text2(fpath.clone(), part_ranges[x].0, part_ranges[x].1)).collect();
    let texts_len:Vec<usize> = texts.iter().enumerate().map(|(i,x)| x.len() - (if i+1 == texts.len() {0} else {hacksize})).collect();

    fn worker(texts:&Vec<Vec<u8>>, starts:Vec<usize>, ends:Vec<usize>, texts_len:Vec<usize>, part:usize,
            merged_dir: String, part_files: Vec<String>, ratio: usize, hacksize: usize) {
        // starts and ends is counting by token

        let nn = texts.len();
        let mut tables:Vec<TableStream> = (0..nn).map(|x| {
            make_table(part_files[x].clone(), starts[x], ratio)
        }).collect();

        let mut idxs:Vec<u64> = starts.iter().map(|&x| x as u64).collect();

        let delta:Vec<u64> = (0..nn).map(|x| {
            let pref:Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect();
            pref.iter().sum::<u64>() - (hacksize * x) as u64
        }).collect();

        let mut next_table = std::io::BufWriter::new(File::create(format!("{}/{:04}", merged_dir.clone(), part)).unwrap());

        fn get_next_maybe_skip(mut tablestream:&mut TableStream,
                               index:&mut u64, thresh:usize) -> u64 {
            let mut location = get_next_pointer_from_table_canfail(&mut tablestream);
            *index += 1;
            if location == u64::MAX {
                return location;
            }
            while location >= thresh as u64 {
                location = get_next_pointer_from_table_canfail(&mut tablestream);
                *index += 1;
                if location == u64::MAX {
                    return location;
                }
            }
            return location;
        }

        let mut heap = BinaryHeap::new();

        for x in 0..nn {
            let position = get_next_maybe_skip(&mut tables[x],
                                               &mut idxs[x], texts_len[x]);
            if idxs[x] <= ends[x] as u64 {
                assert!(position != u64::MAX);
                heap.push(MergeState {
                    suffix: &texts[x][position as usize..],
                    position: position,
                    table_index: x,
                    hacksize: hacksize
                });
            }
        }

        // // Our algorithm is not linear time if there are really long duplicates
        // // found in the merge process. If this happens we'll warn once.
        // let mut did_warn_long_sequences = false;

        let mut prev = &texts[0][0..];
        while let Some(MergeState {suffix: _suffix, position, table_index, hacksize}) = heap.pop() {
            next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()[..ratio]).expect("Write OK");

            let position = get_next_maybe_skip(&mut tables[table_index],
                                               &mut idxs[table_index], texts_len[table_index],);

            if idxs[table_index] <= ends[table_index] as u64 {
                assert!(position != u64::MAX);
                // let next = &texts[table_index][position as usize..];

                // let match_len = (0..(hacksize+1)).find(|&j| !(j < next.len() && j < prev.len() && next[j] == prev[j]));
                // if !did_warn_long_sequences {
                //     if let Some(match_len_) = match_len {
                //         if match_len_ >= hacksize {
                //             println!("There is a match longer than {} bytes.", hacksize);
                //             println!("You probably don't want to be using this code on this dataset---it's (possibly) quadratic runtime now.");
                //             did_warn_long_sequences = true;
                //         }
                //     } else {
                //         println!("There is a match longer than {} bytes.", hacksize);
                //         println!("You probably don't want to be using this code on this dataset---it's quadratic runtime now.");
                //         did_warn_long_sequences = true;
                //     }
                // }

                heap.push(MergeState {
                    suffix: &texts[table_index][position as usize..],
                    position: position,
                    table_index: table_index,
                    hacksize: hacksize
                });
                // prev = next;
            }
        }
    }

    // Make sure we have enough space to take strided offsets for multiple threads
    // This should be an over-approximation, and starts allowing new threads at 1k of data
    let num_threads = std::cmp::min(num_threads, std::cmp::max((texts[0].len() as i64 - 1024)/10, 1));

    // Start a bunch of jobs that each work on non-overlapping regions of the final resulting suffix array
    // Each job is going to look at all of the partial suffix arrays to take the relavent slice.
    let _answer = crossbeam::scope(|scope| {

        let mut tables:Vec<BufReader<File>> = (0..nn).map(|x| {
            std::io::BufReader::new(fs::File::open(part_files[x].clone()).unwrap())
        }).collect();

        let mut starts = vec![0; nn];

        for i in 0..num_threads as usize {
            let texts = &texts;
            let mut ends: Vec<usize> = vec![0; nn];
            if i < num_threads as usize-1 {
                ends[0] = (texts[0].len() / token_width + (num_threads as usize)) / (num_threads as usize) * (i+1);
                let end_seq = &texts[0][table_load_disk(&mut tables[0], ends[0], ratio as usize)..];

                for j in 1..ends.len() {
                    ends[j] = off_disk_position(&texts[j], &mut tables[j], end_seq, ratio as usize, token_width);
                }
            } else {
                for j in 0..ends.len() {
                    ends[j] = texts[j].len() / token_width;
                }
            }

            // println!("Spawn {}: {:?} {:?}", i, starts, ends);

            let starts2 = starts.clone();
            let ends2 = ends.clone();
            let texts_len2 = texts_len.clone();
            let part_files2 = part_files.clone();
            let _one_result = scope.spawn(move || {
                worker(texts,
                       starts2,
                       ends2,
                       texts_len2,
                       i,
                       (*merged_dir).clone(),
                       part_files2,
                       ratio as usize,
                       hacksize
                );
            });

            for j in 0..ends.len() {
                starts[j] = ends[j];
            }
        }
    });

    Ok(())
}

fn cmd_concat(fpath: &String, merged_dir: &String, merged_file: &String, num_threads: i64, ratio: usize, token_width: usize)  -> std::io::Result<()> {

    let ds_size = std::fs::metadata(fpath.clone()).unwrap().len();

    // on-disk files to store concat'ed sa
    let merged_buf = OpenOptions::new().write(true).create(true).open(merged_file).unwrap();
    let merged_buf_size = ds_size / token_width as u64 * ratio as u64;
    merged_buf.set_len(merged_buf_size).unwrap();
    drop(merged_buf);

    // this worker writes to concat'ed sa to disk
    fn cat_worker(part: usize, offset: u64, ratio: usize, merged_dir: String, merged_file: String) {
        let mut buffer = vec![0u8; ratio*1024*1024];

        let merged_part_file = format!("{}/{:04}", merged_dir, part);
        let mut merged_part = std::io::BufReader::new(fs::File::open(merged_part_file).unwrap());
        let mut merged = std::io::BufWriter::new(OpenOptions::new().write(true).open(merged_file).unwrap());
        let merged_offset = (ratio as u64) * offset;
        merged.seek(SeekFrom::Start(merged_offset)).expect("Seek failed!");
        while let Ok(n) = merged_part.read(&mut buffer) {
            assert!(n % ratio == 0);
            if n == 0 { break; }
            merged.write_all(&buffer[..n]).expect("Write OK");
        }
    }

    let _answer = crossbeam::scope(|scope| {
        let mut offset = 0;
        for i in 0..num_threads as usize {
            let _one_result = scope.spawn(move || {
                cat_worker(
                    i,
                    offset,
                    ratio,
                    (*merged_dir).clone(),
                    (*merged_file).clone()
                );
            });
            offset += std::fs::metadata(format!("{}/{:04}", merged_dir, i)).unwrap().len() / ratio as u64;
        }
    });

    Ok(())
}

fn fnv1a(key: &[u8], seed: u64) -> u64 {
    let mut h = seed;
    for &b in key {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3u64);
    }
    h
}

fn bloom_insert(data: &mut [u8], num_bits: u64, num_hashes: u32, key: &[u8]) {
    let h1 = fnv1a(key, 0xcbf29ce484222325u64);
    let h2 = fnv1a(key, 0x517cc1b727220a95u64);
    for i in 0..num_hashes as u64 {
        let bit = h1.wrapping_add(i.wrapping_mul(h2)) % num_bits;
        data[(bit / 8) as usize] |= 1 << (bit % 8);
    }
}

fn cmd_build_bloom(data_file: &String, bloom_file: &String,
                   num_bits: u64, num_hashes: u32, max_ngram_n: u32,
                   token_width: usize) -> std::io::Result<()> {
    let now = Instant::now();
    println!("Building bloom filter for {}", data_file);

    let ds = filebuffer::FileBuffer::open(data_file)?;
    let tok_cnt = ds.len() / token_width;

    let num_bytes = ((num_bits + 7) / 8) as usize;
    let mut bloom_data = vec![0u8; num_bytes];

    // doc_sep is all 0xFF bytes of token_width length
    let doc_sep: Vec<u8> = vec![0xffu8; token_width];

    let mut tokens_since_sep: u64 = 0;

    for pos in 0..tok_cnt {
        let byte_offset = pos * token_width;
        let token_bytes = &ds[byte_offset..byte_offset + token_width];

        if token_bytes == &doc_sep[..] {
            tokens_since_sep = 0;
            continue;
        }
        tokens_since_sep += 1;

        for n in 2..=max_ngram_n {
            if tokens_since_sep >= n as u64 {
                let start = (pos + 1 - n as usize) * token_width;
                let end = (pos + 1) * token_width;
                bloom_insert(&mut bloom_data, num_bits, num_hashes, &ds[start..end]);
            }
        }
    }

    // Write: header (num_bits: 8 bytes, num_hashes: 4 bytes, max_ngram_n: 4 bytes) + data
    let mut out = File::create(bloom_file)?;
    out.write_all(&num_bits.to_le_bytes())?;
    out.write_all(&num_hashes.to_le_bytes())?;
    out.write_all(&max_ngram_n.to_le_bytes())?;
    out.write_all(&bloom_data)?;

    println!("Bloom filter built in {:.2}s, {} bytes written to {}",
             now.elapsed().as_secs_f64(), 16 + bloom_data.len(), bloom_file);
    Ok(())
}

/*
 * Build a 2-gram index that maps every unique bigram (pair of consecutive tokens)
 * to its [lo, hi) range in the suffix array. At query time, this allows the engine
 * to skip the initial binary search over the entire SA and instead start from the
 * narrowed range for the query's first 2 tokens.
 *
 * File format:
 *   Header: num_entries (u64), token_width (u32), padding (u32)
 *   Entries: bigram_key (u64), lo (u64), hi (u64)
 *
 * The bigram_key is the raw 2*token_width bytes from the data file packed into a u64
 * (zero-extended, little-endian). This matches how the C++ engine constructs keys
 * from query token IDs via memcpy.
 */
fn cmd_build_bigram(data_file: &String, table_file: &String, bigram_file: &String,
                    token_width: usize, ratio: usize) -> std::io::Result<()> {
    let now = Instant::now();
    println!("Building bigram index for {}", data_file);

    let ds = filebuffer::FileBuffer::open(data_file)?;
    let sa = filebuffer::FileBuffer::open(table_file)?;

    let ds_size = ds.len();
    let tok_cnt = ds_size / token_width;
    let bigram_bytes = 2 * token_width;

    assert_eq!(sa.len(), tok_cnt * ratio, "SA size mismatch: expected {} * {} = {}, got {}",
               tok_cnt, ratio, tok_cnt * ratio, sa.len());

    // We stream entries to disk to avoid holding them all in memory
    let mut out = std::io::BufWriter::new(File::create(bigram_file)?);

    // Write placeholder header
    let placeholder_num_entries: u64 = 0;
    out.write_all(&placeholder_num_entries.to_le_bytes())?;
    let tw = token_width as u32;
    out.write_all(&tw.to_le_bytes())?;
    let padding = 0u32;
    out.write_all(&padding.to_le_bytes())?;

    let mut num_entries: u64 = 0;
    let mut prev_key: u64 = u64::MAX; // sentinel: no current group
    let mut cur_lo: u64 = 0;

    for rank in 0..tok_cnt {
        // Read suffix array pointer for this rank
        let mut ptr_bytes = [0u8; 8];
        ptr_bytes[..ratio].copy_from_slice(&sa[rank * ratio..rank * ratio + ratio]);
        let ptr = u64::from_le_bytes(ptr_bytes) as usize;

        // Check if suffix is long enough for a 2-gram
        if ptr + bigram_bytes > ds_size {
            // Suffix too short — finalize current group if any
            if prev_key != u64::MAX {
                out.write_all(&prev_key.to_le_bytes())?;
                out.write_all(&cur_lo.to_le_bytes())?;
                out.write_all(&(rank as u64).to_le_bytes())?;
                num_entries += 1;
                prev_key = u64::MAX;
            }
            continue;
        }

        // Pack the 2-gram bytes into a u64 key
        let mut key_bytes = [0u8; 8];
        key_bytes[..bigram_bytes].copy_from_slice(&ds[ptr..ptr + bigram_bytes]);
        let key = u64::from_le_bytes(key_bytes);

        if key != prev_key {
            // Finalize previous group
            if prev_key != u64::MAX {
                out.write_all(&prev_key.to_le_bytes())?;
                out.write_all(&cur_lo.to_le_bytes())?;
                out.write_all(&(rank as u64).to_le_bytes())?;
                num_entries += 1;
            }
            prev_key = key;
            cur_lo = rank as u64;
        }
    }

    // Finalize last group
    if prev_key != u64::MAX {
        out.write_all(&prev_key.to_le_bytes())?;
        out.write_all(&cur_lo.to_le_bytes())?;
        out.write_all(&(tok_cnt as u64).to_le_bytes())?;
        num_entries += 1;
    }

    out.flush()?;
    drop(out);

    // Go back and write the actual num_entries in the header
    let mut header_out = OpenOptions::new().write(true).open(bigram_file)?;
    header_out.write_all(&num_entries.to_le_bytes())?;

    println!("Bigram index built in {:.2}s, {} unique 2-grams written to {}",
             now.elapsed().as_secs_f64(), num_entries, bigram_file);
    Ok(())
}

fn main()  -> std::io::Result<()> {

    let args = Args::parse();

    match &args.command {
        Commands::MakePart { data_file, parts_dir, start_byte, end_byte, ratio, token_width } => {
            cmd_make_part(data_file, parts_dir, *start_byte as u64, *end_byte as u64, *ratio, *token_width)?;
        }

        Commands::Merge { data_file, parts_dir, merged_dir, num_threads, hacksize, ratio, token_width } => {
            cmd_merge(data_file, parts_dir, merged_dir, *num_threads, *hacksize, *ratio, *token_width)?;
        }

        Commands::Concat { data_file, merged_dir, merged_file, num_threads, ratio, token_width } => {
            cmd_concat(data_file, merged_dir, merged_file, *num_threads, *ratio, *token_width)?;
        }

        Commands::BuildBloom { data_file, bloom_file, num_bits, num_hashes, max_ngram_n, token_width } => {
            cmd_build_bloom(data_file, bloom_file, *num_bits, *num_hashes, *max_ngram_n, *token_width)?;
        }

        Commands::BuildBigram { data_file, table_file, bigram_file, token_width, ratio } => {
            cmd_build_bigram(data_file, table_file, bigram_file, *token_width, *ratio)?;
        }
    }

    Ok(())
}
