use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use plotters::prelude::*;

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

    let mut points : Vec<(f64, f64)> = Vec::new();
    let mut ipoints : Vec<(i32, i32)> = Vec::new();

    for n in 0..temp.len() {
        points.push((temp[n] * 1_000.0, humi[n] * 1_000.0));
        ipoints.push(((temp[n] * 1_000.0) as i32, (humi[n] * 1_000.0) as i32));
    }

    //println!("{:#?}", ipoints);

        // Create a new plotting backend
    let root = BitMapBackend::new("output.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Plot the points
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d( -20_000..40__000, 0..1_000)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let line: Vec<(i32, i32)> = (-10..=40).map(|x| ( x * 1_000, ((&res.1 + &res.0 * (x as f64)) * 1_000.0) as i32)).collect();
    //println!("{:#?}", line);

    chart
        .draw_series(ipoints.into_iter()
            .map(|point| Circle::new(point, 1, &RED)))
        .unwrap();

    chart
    .draw_series(
        LineSeries::new(line, &GREEN)
        ).unwrap();

    // Save the image to a file
    root.present().unwrap();

    timestamp("Drawing");

    Ok(())
}
