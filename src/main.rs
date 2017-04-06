extern crate autodiff;
extern crate num;
extern crate gnuplot;

use autodiff::AutoDiff;
use std::io::{self,Write};
use num::Float;
use gnuplot::{Figure, Caption, Color, LegendOption, Coordinate};
use std::f64::consts::PI;

// A function `fn plot_fn_with_derivative<T, F>(func: F...) where F: Fn(T) -> T`
// does not work, since calls are monomorphized at call site, meaning that
// F can not be generic when plot_fn_with_derivative is monomorphized.
macro_rules! plot_fn_with_derivative {
    ($func:ident, ($start:expr, $end:expr, $incr:expr), $caption_fx:expr, $caption_dfx:expr) => {{
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut dys = Vec::new();

        let start: f64 = $start as f64;
        let stop: f64 = $end as f64;
        let incr: f64 = $incr as f64;

        let mut x = start;
        while x <= stop {
            let ad = AutoDiff::var(x);
            let sig = $func(ad);
            xs.push(x);
            ys.push(sig.val());
            dys.push(sig.dval());
            x += incr;
        }

        let mut fig = Figure::new();
        fig.axes2d().lines(
            &xs,
            &ys,
            &[Caption($caption_fx), Color("blue")]
        ).lines(
            &xs,
            &dys,
            &[Caption($caption_dfx), Color("black")]
        ).set_legend(
            Coordinate::Graph(0.95f64),
            Coordinate::Graph(0.95f64),
            &[], &[]
        );
        fig.show();
    }};
}


fn main() {
    plot_fn_with_derivative!(sigmoid,
                             (-10f64, 10f64, 0.1),
                             "f(x) = 1/(1+exp(-x))",
                             "f'(x) = f(x)(1 - f(x))");

    plot_fn_with_derivative!(e_to_pi_x,
                             (0f64, 10f64, 0.1),
                             "f(x) = exp(pi*x)",
                             "f'(x) = pi * exp(pi*x)");

    plot_fn_with_derivative!(sin_2x,
                             (0, 2f64 * PI, 0.1),
                             "f(x) = 2sinθcosθ = sin(2θ)",
                             "f'(x) = 2cos(2θ)");

}

/// computes exp(pi*x)
fn e_to_pi_x<T>(x: T) -> T
    where T: Float + From<f64>
{
    let pi: T = From::from(std::f64::consts::PI);
    T::exp(pi * x)
}

/// computes `1/(1 + exp(-x))`
fn sigmoid<T>(x: T) -> T
    where T: Float
{
    T::one() / (T::one() + (-x).exp())
}

/// computes sin(2*x)
fn sin_2x<T>(x: T) -> T where T: Float + From<f64> {
    <T as From<f64>>::from(2f64) * x.sin() * x.cos()
}

// A_c * cos(w_c*t - A_m * sin(w_m * t))
fn fm<T>(t: T) -> T where T: Float + From<f64> {
    let A_c: T = From::from(5f64);
    let A_m: T = From::from(2f64);
    let f_c: T = From::from(4f64); // Hz
    let f_m: T = From::from(1.5f64); // Hz
    let pi: T = From::from(std::f64::consts::PI);
    let two: T = From::from(2f64);
    A_c * (two * pi * f_c * t - A_m * (two * pi * t).sin()).cos()
}



