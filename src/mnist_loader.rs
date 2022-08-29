// Source: https://ngoldbaum.github.io/posts/loading-mnist-data-in-rust/
use byteorder::{BigEndian, ReadBytesExt};
use ndarray::Array2;
use std::{
    fs::File,
    io::{Cursor, Read},
};

#[derive(Debug)]
pub struct MnistImage {
    pub image: Array2<f64>, // 0.0 means background white and 1.0 means black.
    pub classification: u8, // Label values are 0 to 9.
}

/// Load an MNIST dataset and returns a vector containing the MnistImages.
pub fn load_data(dataset_name: &str) -> Result<Vec<MnistImage>, std::io::Error> {
    // Load the label data.
    let filename = format!("{}-labels-idx1-ubyte.gz", dataset_name);
    let label_data = &MnistData::new(&(File::open(filename))?)?;

    // Load the image data.
    let filename = format!("{}-images-idx3-ubyte.gz", dataset_name);
    let images_data = &MnistData::new(&(File::open(filename))?)?;

    // Load
    let mut images: Vec<Array2<f64>> = Vec::new();
    let image_shape = (images_data.sizes[1] * images_data.sizes[2]) as usize;
    for i in 0..images_data.sizes[0] as usize {
        // Get slice of image data.
        let start = i * image_shape;
        let image_data = images_data.data[start..start + image_shape].to_vec();

        // Convert greyscale values to floats from 0.0 to 1.0.
        let image_data: Vec<f64> = image_data.into_iter().map(|x| x as f64 / 255.0).collect();

        // Convert image data to a column vector.
        images.push(Array2::from_shape_vec((image_shape, 1), image_data).unwrap());
    }

    let classifications: Vec<u8> = label_data.data.clone();

    // Return a vector containing `MnistImage`s, each containg the column vector storing the image data and it's label.
    let mut ret: Vec<MnistImage> = Vec::new();
    for (image, classification) in images.into_iter().zip(classifications.into_iter()) {
        ret.push(MnistImage {
            image,
            classification,
        });
    }

    Ok(ret)
}

#[derive(Debug)]
struct MnistData {
    sizes: Vec<i32>,
    data: Vec<u8>,
}

impl MnistData {
    fn new(f: &File) -> Result<MnistData, std::io::Error> {
        // Create a new dedcoder from the file.
        let mut gz = flate2::read::GzDecoder::new(f);

        // Read all of the data and place in the buffer.
        let mut contents: Vec<u8> = Vec::new();
        gz.read_to_end(&mut contents)?;

        // Create a cursor for the data.
        let mut r = Cursor::new(&contents);

        let magic_number = r.read_i32::<BigEndian>()?;

        let mut sizes: Vec<i32> = Vec::new();
        let mut data: Vec<u8> = Vec::new();

        // idx file format descirbed here: https://deepai.org/dataset/mnist
        match magic_number {
            // Labels
            2049 => sizes.push(r.read_i32::<BigEndian>()?), // # of items
            // Images
            2051 => {
                sizes.push(r.read_i32::<BigEndian>()?); // # of images
                sizes.push(r.read_i32::<BigEndian>()?); // # of rows
                sizes.push(r.read_i32::<BigEndian>()?); // # of columns
            }
            _ => panic!(),
        }

        // Load the data (either image data or label data).
        r.read_to_end(&mut data)?;

        Ok(MnistData { sizes, data })
    }
}
