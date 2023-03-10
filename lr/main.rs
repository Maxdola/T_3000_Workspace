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
    let mut ref_time = Instant::now();

    let mut timestamp = move | name: &str | {
        println!("[{:?}]: --- {:?} ms ---", name, ref_time.elapsed().as_millis());
        ref_time = Instant::now();
    };

    timestamp("Imports");
    
    let path = Path::new("../data/weatherHistory.csv");
    //let path = Path::new("../data/weatherHistory_big.csv");
    /*let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;*/
    let file = File::open(path)?;
    let mut buf_reader = std::io::BufReader::new(file);
    let mut contents = String::new();
    buf_reader.read_to_string(&mut contents)?;
    timestamp("Reading File");

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

    timestamp("Data Digestion");

    //println!("{:?}", temp.len());
    //println!("{:?}", humi.len());

    let result = linear_regression::<f64,f64,f64>(&temp, &humi);

    let res = result.unwrap();

    timestamp("Training");


    let x0 = predict(0.0, &res);
    let x2 = predict(0.0, &res);

    timestamp("Prediction");

    println!("{:?}", res);

    println!("Predicted value for '0': {:?}", x0);
    println!("Predicted value for '2': {:?}", x2);
    

    println!("Time elapsed: {:?} ms", start.elapsed().as_millis());


    Ok(())
}
