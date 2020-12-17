use ndarray::prelude::*;

fn update(x: ArrayView1<u32>, u: ArrayView1<u32>) -> std::io::Result<()> {
    Ok(())
}

fn predict() -> std::io::Result<()> {
    Ok(())
}

fn main() {
    let a = array![1, 2, 3];
    let b = array![4, 5, 6];
    let c = update(a.view(), b.view());

    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", c);
}
