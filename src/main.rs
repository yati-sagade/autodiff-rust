extern crate autodiff;
extern crate num;
extern crate gnuplot;

use autodiff::AutoDiff;
use num::Float;
use gnuplot::{Figure,Caption,Color};


fn sigmoid<T>(x: T) -> T where T: Float {
    T::one() / (T::one() + (-x).exp())
}

fn main() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for x in -10..10 {
        println!("{}", sigmoid(x as f64));
        let ad = AutoDiff::var(x as f64);
        let sig = sigmoid(ad);
        xs.push(x);
        ys.push(sig.dval());
    }
    let mut fig = Figure::new();
    fig.axes2d().lines(&xs, &ys, &[Caption("Derivative of the sigmoid"), Color("black")]);
    fig.show();
}
