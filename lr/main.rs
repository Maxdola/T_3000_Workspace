use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use linreg::{linear_regression};

use std::time::{Instant};

fn predict(x: f64, result: &(f64, f64)) -> f64 {
    result.1 + result.0 * x
}

fn main() -> Result<(), Box<dyn Error>> {
    let start = Instant::now();

    //let path = Path::new("./weatherHistory.csv");
    let path = Path::new("./weatherHistory_big.csv");
    /*let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;*/
    let file = File::open(path)?;
    let mut buf_reader = std::io::BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;

    let mut temp : Vec<f64> = Vec::new();
    let mut humi : Vec<f64> = Vec::new();

    let header_line = contents.lines().next().unwrap();
    let header_columns: Vec<&str> = header_line.split(',').collect();
    let header_map: HashMap<&str, usize> = header_columns
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i))
        .collect();
    //println!("{:?}", header_map);
    let a_index = header_map.get("Temperature (C)").unwrap();
    let b_index = header_map.get("Humidity").unwrap();

    for line in contents.lines().skip(1) {
        let columns: Vec<&str> = line.split(',').collect();
        if let (Ok(a), Ok(b)) = (
            columns[*a_index].parse::<f64>(),
            columns[*b_index].parse::<f64>(),
        ) {
            temp.push(a);
            humi.push(b);
        }
    }

    //println!("{:?}", temp.len());
    //println!("{:?}", humi.len());

    let result = linear_regression::<f64,f64,f64>(&temp, &humi);

    let res = result.unwrap();

    println!("{:?}", res);

    println!("Predicted value for '0': {:?}", predict(0.0, &res));
    println!("Predicted value for '2': {:?}", predict(2.0, &res));
    
    let duration = start.elapsed();

    println!("Time elapsed: {:?}ms", duration.as_millis());

    Ok(())
}
