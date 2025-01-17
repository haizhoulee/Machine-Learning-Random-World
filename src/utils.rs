
//! Utility routines for loading and storing data into files.
use ndarray::prelude::*;
use std::fmt::Display;
use std::error::Error;
use std::fs::{File, OpenOptions};
use csv::{ReaderBuilder, WriterBuilder};

/// Loads a CSV data file.
///
/// The file format should be, for each row:
///     label, x1, x2, ...
/// where x1, x2, ... are features forming a feature vector.
pub fn load_data(fname: &str) -> Result<(Array2<f64>, Array1<usize>), Box<Error>> {
    let mut reader = ReaderBuilder::new()
                                   .has_headers(false)
                                   .from_path(fname)?;

    let mut inputs: Vec<f64> = Vec::new();
    let mut targets: Vec<usize> = Vec::new();

    let mut d: Option<usize> = None;

    for result in reader.records() {
        let record = result?;

        inputs.extend(record.iter()
                            .skip(1)  // First one is the label.
                            .map(|x| x.trim()
                                      .parse::<f64>().ok()
                                                     .expect("Failed to parse")));
        targets.push(record[0].parse::<usize>()?);

        if let Some(x) = d {
            if x != record.len() - 1 {
                panic!("File has wrong format");
            }
        } else {
            d = Some(record.len() - 1);
        }
    }

    let inputs_a = if let Some(d) = d {
        let n = inputs.len() / d;
        Array::from_vec(inputs)
              .into_shape((n, d))?
    } else {
        panic!("File has wrong format");
    };

    Ok((inputs_a, Array::from_vec(targets)))
}

/// Stores predictions into a CSV file.
///
/// It stores either predictions (bool values) or p-values (f64)
/// into a CSV file. Each line contains the predictions/p-values
/// for one test object:
///     x1, x2, ...
/// where each value corresponds to a label.
pub fn store_predictions<T>(predictions: ArrayView2<T>, fname: &str,
        append: bool) -> Result<(), Box<Error>> where T: Display {
    if !append {
        // Empty the file.
        let _ = File::create(fname)
                     .expect("Failed to create file");
    }
    let mut writer = WriterBuilder::new().from_writer(OpenOptions::new()
                                             .append(append)
                                             .write(true)
                                             .open(fname)?);

    for x in predictions.outer_iter() {
        writer.write_record(x.iter()
                             .map(|v| format!("{}", v)))?;
    }

    writer.flush()?;
    Ok(())
}

/// Loads predictions into a CSV file.
///
/// It stores either predictions (bool values) or p-values (f64)
/// into a CSV file. Each line contains the predictions/p-values
/// for one test object:
///     x1, x2, ...
/// where each value corresponds to a label.
pub fn load_pvalues(fname: &str) -> Result<Array2<f64>, Box<Error>> {
    let mut reader = ReaderBuilder::new()
                                   .has_headers(false)
                                   .from_path(fname)?;

    let mut pvalues = vec![];
    let mut d: Option<usize> = None;

    for result in reader.records() {
        let record = result?;

        pvalues.extend(record.iter()
                             .map(|x| x.trim()
                                       .parse::<f64>()
                                       .ok()
                                       .expect("Failed to parse")));
        // Update length of a row.
        if let Some(d) = d {
            if d != record.len() {
                panic!("File has wrong format");
            }
        } else {
            d = Some(record.len());
        }
    }

    let pvalues_a = if let Some(d) = d {
        let n = pvalues.len() / d;
        Array::from_vec(pvalues)
              .into_shape((n, d))?
    } else {
        panic!("File has wrong format");
    };

    Ok(pvalues_a)
}