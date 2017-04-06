# Autodiff

[Forward accumulation based automatic differentiation for Rust][1]

## Example

This is a small program that has a function to compute the derivatives
of some functions and plot them using gnuplot.

    // main.rs

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


That program produces the following plots:

![plot for the sigmoid and its derivative][img/sigmoid.png]
![plot for exp(pi*x) and its derivative][img/exp.png]
![plot for sin(2x) and its derivative][img/sin.png]


[1]: https://en.wikipedia.org/wiki/Automatic_differentiation#Forward_accumulation
