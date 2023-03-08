use pyo3::prelude::*;
use std::collections::HashMap;
//use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use linreg::{linear_regression};

#[pyfunction]
fn read_csv(file_path : &str, cols: Vec<&str>) -> PyResult<Vec<Vec<f64>>> {

    let path = Path::new(file_path);
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut col_values : Vec<Vec<f64>> = Vec::new();
    for i in 0..cols.len()  {
        col_values.push(Vec::new());
    }

    let header_line = contents.lines().next().unwrap();
    let header_columns: Vec<&str> = header_line.split(',').collect();
    let header_map: HashMap<&str, usize> = header_columns
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();

    let mut col_indexes : Vec<&usize> = Vec::new();

    for col in cols {
        let index = header_map.get(col).unwrap();
        col_indexes.push(index);
    }

    for line in contents.lines().skip(1) {
        let columns: Vec<&str> = line.split(',').collect();

        
        for i in 0..col_indexes.len() {
            let col_index = col_indexes[i];
            let value = columns[*col_index].parse::<f64>().unwrap();
            col_values[i].push(value);
        }
    }

    Ok(col_values)
}

#[pyfunction]
fn train(x: Vec<f64>, y: Vec<f64>) -> (f64, f64) {
    let result = linear_regression::<f64,f64,f64>(&x, &y);
    let res = result.unwrap();
    return (res.0, res.1);
}

#[pyfunction]
fn predict(model: (f64, f64), value: f64) -> f64 {
    return model.1 + model.0 * value;
}

#[pymodule]
fn QyO3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_csv, m)?)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    Ok(())
}