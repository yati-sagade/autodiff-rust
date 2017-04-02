extern crate num;

use std::fmt;
use std::ops::{Add, Sub, Div, Mul, Rem, Neg};
use std::convert::From;
use num::{Float, Num, Zero, One};
use num::cast::{ToPrimitive, NumCast};
use std::cmp::{PartialOrd, Ordering};
use std::num::FpCategory;

macro_rules! assert_almost_equal {
    ($a:expr, $b:expr) => {
        if ($a-$b).abs() > 1e-6 {
            panic!(format!("left: {}, right: {} <- not almost the same.", $a, $b));
        } else {
            assert!(true);
        }
    };
}


#[derive(Copy,Clone)]
pub struct AutoDiff {
    /// Current value of the function.
    val: f64,
    /// Current derivative value.
    dval: f64,
}

impl AutoDiff {
    pub fn new(val: f64, dval: f64) -> AutoDiff {
        AutoDiff {
            val: val,
            dval: dval,
        }
    }

    pub fn val(&self) -> f64 {
        self.val
    }

    pub fn dval(&self) -> f64 {
        self.dval
    }

    pub fn var(val: f64) -> AutoDiff {
        AutoDiff::new(val, 1f64)
    }

    pub fn constant(val: f64) -> AutoDiff {
        From::from(val)
    }
}

impl fmt::Display for AutoDiff {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.val, self.dval)
    }
}

impl Neg for AutoDiff {
    type Output = AutoDiff;
    fn neg(self) -> AutoDiff {
        AutoDiff::new(self.val.neg(), self.dval.neg())
    }
}

impl ToPrimitive for AutoDiff {
    fn to_i64(&self) -> Option<i64> {
        self.val.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.val.to_u64()
    }
}

impl NumCast for AutoDiff {
    fn from<T>(n: T) -> Option<AutoDiff>
        where T: ToPrimitive
    {
        // We first try to cast T to an f64, and then use this
        // to initialize a constant AutoDiff value.
        NumCast::from(n).map(<AutoDiff as From<f64>>::from)
    }
}

impl Zero for AutoDiff {
    fn zero() -> AutoDiff {
        From::from(f64::zero())
    }
    fn is_zero(&self) -> bool {
        self.val.is_zero()
    }
}

impl Num for AutoDiff {
    type FromStrRadixErr = num::traits::ParseFloatError;
    fn from_str_radix(s: &str, radix: u32) -> Result<AutoDiff, num::traits::ParseFloatError> {
        f64::from_str_radix(s, radix).map(From::from)
    }
}

// Not sure about the derivative, so not doing anything.
// http://math.stackexchange.com/questions/672610/derivative-of-remainder-function
impl Rem for AutoDiff {
    type Output = AutoDiff;
    fn rem(self, rhs: AutoDiff) -> AutoDiff {
        From::from(self.val.rem(rhs.val))
    }
}

impl One for AutoDiff {
    fn one() -> AutoDiff {
        From::from(f64::one())
    }
}

impl PartialEq for AutoDiff {
    fn eq(&self, other: &AutoDiff) -> bool {
        self.val.eq(&other.val)
    }
}

impl PartialOrd for AutoDiff {
    fn partial_cmp(&self, other: &AutoDiff) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl Float for AutoDiff {
    fn nan() -> AutoDiff {
        From::from(f64::nan())
    }
    fn infinity() -> AutoDiff {
        From::from(f64::infinity())
    }
    fn neg_infinity() -> AutoDiff {
        From::from(f64::neg_infinity())
    }
    fn neg_zero() -> AutoDiff {
        From::from(f64::neg_zero())
    }
    fn min_value() -> AutoDiff {
        From::from(f64::min_value())
    }
    fn min_positive_value() -> AutoDiff {
        From::from(f64::min_positive_value())
    }
    fn max_value() -> AutoDiff {
        From::from(f64::max_value())
    }
    fn is_nan(self) -> bool {
        self.val.is_nan()
    }
    fn is_infinite(self) -> bool {
        self.val.is_infinite()
    }
    fn is_finite(self) -> bool {
        self.val.is_finite()
    }
    fn is_normal(self) -> bool {
        self.val.is_normal()
    }
    fn classify(self) -> FpCategory {
        self.val.classify()
    }
    fn floor(self) -> AutoDiff {
        From::from(self.val.floor())
    }
    fn ceil(self) -> AutoDiff {
        From::from(self.val.ceil())
    }
    fn round(self) -> AutoDiff {
        From::from(self.val.round())
    }
    fn trunc(self) -> AutoDiff {
        From::from(self.val.trunc())
    }
    fn fract(self) -> AutoDiff {
        From::from(self.val.fract())
    }
    fn abs(self) -> AutoDiff {
        From::from(self.val.abs())
    }
    fn signum(self) -> AutoDiff {
        From::from(self.val.signum())
    }
    fn is_sign_positive(self) -> bool {
        self.val.is_sign_positive()
    }
    fn is_sign_negative(self) -> bool {
        self.val.is_sign_negative()
    }
    fn mul_add(self, a: AutoDiff, b: AutoDiff) -> AutoDiff {
        AutoDiff::new(self.val.mul_add(a.val, b.val),
                      self.val * a.dval + self.dval * a.val + b.dval /* D(x*y+z) = x*Dy + y*Dx + Dz */)
    }
    fn recip(self) -> AutoDiff {
        let r = self.val.recip();
        AutoDiff::new(r, -r * r * self.dval) // D(1/x) = -Dx/(x*x)
    }
    fn powi(self, n: i32) -> AutoDiff {
        let p = self.val.powi(n);
        let d = self.dval * (n as f64) * p / self.val; // D(x^n) = n * (x^(n-1)) Dx =  n * (x^k) / x * Dx
        AutoDiff::new(p, d)
    }
    fn powf(self, n: AutoDiff) -> AutoDiff {
        let p = self.val.powf(n.val);
        let d = self.dval * n.val * p / self.val; // D(x^n) = n * (x^(n-1)) Dx =  n * (x^k) / x * Dx
        AutoDiff::new(p, d)
    }
    fn sqrt(self) -> AutoDiff {
        AutoDiff::new(self.val.sqrt(),
                      0.5 * self.dval / self.val.sqrt() /* D(sqrt(x)) = D(x^(1/2)) =  Dx / (2 * sqrt(x)) */)
    }
    fn exp(self) -> AutoDiff {
        AutoDiff::new(self.val.exp(), self.val.exp() * self.dval)
    }
    fn exp2(self) -> AutoDiff {
        let p = self.val.exp2();
        AutoDiff::new(p, p * 2f64.ln() * self.dval) // D(a^x) = a^x * ln(a) * Dx
    }
    fn ln(self) -> AutoDiff {
        AutoDiff::new(self.val.ln(), self.dval / self.val) // D(ln(x)) = Dx/x
    }
    fn log(self, base: AutoDiff) -> AutoDiff {
        // D(log_a(x)) = Dx / (x * ln(a))
        AutoDiff::new(self.val.log(base.val),
                      self.dval / (self.val * base.val.ln()))
    }
    fn log2(self) -> AutoDiff {
        AutoDiff::new(self.val.log2(), self.dval / (self.val * 2f64.ln()))
    }
    fn log10(self) -> AutoDiff {
        AutoDiff::new(self.val.log10(), self.dval / (self.val * 10f64.ln()))
    }
    fn max(self, other: AutoDiff) -> AutoDiff {
        if self.val < other.val {
            other
        } else {
            self
        }
    }
    fn min(self, other: AutoDiff) -> AutoDiff {
        if self.val > other.val {
            other
        } else {
            self
        }
    }
    fn abs_sub(self, other: AutoDiff) -> AutoDiff {
        let d = self - other;
        if d.val < 0f64 {
            -d
        } else {
            d
        }
    }
    fn cbrt(self) -> AutoDiff {
        AutoDiff::new(self.val.cbrt(),
                      self.dval * self.val.powf(-2f64 / 3f64) / 3f64)
    }
    fn hypot(self, other: AutoDiff) -> AutoDiff {
        let p = self.val.hypot(other.val);
        // D(sqrt(x^2+y^2)) = (x*Dx + y*Dy) / sqrt(x^2 + y^2)
        let d = (self.val * self.dval + other.val * other.dval) /
                (self.val.powi(2) + other.val.powi(2)).sqrt();
        AutoDiff::new(p, d)
    }
    fn sin(self) -> AutoDiff {
        AutoDiff::new(self.val.sin(), self.dval * self.val.cos())
    }
    fn cos(self) -> AutoDiff {
        AutoDiff::new(self.val.cos(), -self.dval * self.val.sin())
    }
    fn tan(self) -> AutoDiff {
        AutoDiff::new(self.val.tan(), self.dval / self.val.cos().powi(2))
    }
    fn asin(self) -> AutoDiff {
        AutoDiff::new(self.val.asin(), self.dval / (1f64 - self.val.powi(2)))
    }
    fn acos(self) -> AutoDiff {
        AutoDiff::new(self.val.acos(), -self.dval / (1f64 - self.val.powi(2)))
    }
    fn atan(self) -> AutoDiff {
        AutoDiff::new(self.val.atan(), self.dval / (1f64 + self.val.powi(2)))
    }
    fn atan2(self, y: AutoDiff) -> AutoDiff {
        // D(atan(y/x)) = (xDy - yDx) / (x^2 + y^2)
        let d = (self.val * y.dval - y.val * self.dval) / (self.val.powi(2) + y.val.powi(2));
        AutoDiff::new(self.val.atan2(y.val), self.dval * d)
    }
    fn sin_cos(self) -> (AutoDiff, AutoDiff) {
        (self.sin(), self.cos())
    }
    fn exp_m1(self) -> AutoDiff {
        AutoDiff::new(self.val.exp_m1(), self.dval * self.val.exp())
    }
    fn ln_1p(self) -> AutoDiff {
        AutoDiff::new(self.val.ln_1p(), self.dval / (1f64 + self.val))
    }
    fn sinh(self) -> AutoDiff {
        AutoDiff::new(self.val.sinh(), self.dval * self.val.cosh())
    }
    fn cosh(self) -> AutoDiff {
        AutoDiff::new(self.val.cosh(), self.dval * self.val.sinh())
    }
    fn tanh(self) -> AutoDiff {
        AutoDiff::new(self.val.tanh(),
                      self.dval * (1f64 - self.val.tanh().powi(2)))
    }
    fn asinh(self) -> AutoDiff {
        AutoDiff::new(self.val.asinh(),
                      self.dval / (1f64 + self.val.powi(2)).sqrt())
    }
    fn acosh(self) -> AutoDiff {
        AutoDiff::new(self.val.acosh(),
                      self.dval / ((self.val + 1f64) * (self.val - 1f64)).sqrt())
    }
    fn atanh(self) -> AutoDiff {
        AutoDiff::new(self.val.atanh(), self.dval / (1f64 - self.val.powi(2)))
    }
    fn integer_decode(self) -> (u64, i16, i8) {
        Float::integer_decode(self.val)
    }
}

impl Add for AutoDiff {
    type Output = AutoDiff;
    fn add(self, rhs: AutoDiff) -> AutoDiff {
        AutoDiff::new(self.val + rhs.val, self.dval + rhs.dval)
    }
}


impl Sub for AutoDiff {
    type Output = AutoDiff;
    fn sub(self, rhs: AutoDiff) -> AutoDiff {
        AutoDiff::new(self.val - rhs.val, self.dval - rhs.dval)
    }
}

impl Div for AutoDiff {
    type Output = AutoDiff;
    fn div(self, rhs: AutoDiff) -> AutoDiff {
        AutoDiff::new(self.val / rhs.val,
                      (self.dval * rhs.val - rhs.dval * self.val) / rhs.val.powi(2))
    }
}

// AutoDiff * AutoDiff
impl Mul for AutoDiff {
    type Output = AutoDiff;
    fn mul(self, rhs: AutoDiff) -> AutoDiff {
        AutoDiff::new(self.val * rhs.val,
                      self.val * rhs.dval + self.dval * rhs.val)
    }
}


impl From<f64> for AutoDiff {
    fn from(u: f64) -> AutoDiff {
        AutoDiff {
            val: u,
            dval: 0f64,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    // The breakdown is simply to group kinds of functions we test autodiff
    // against. In each case, we shall define a few functions with known
    // closed form of the derivative, and compare this value with the value
    // computed by autodiff


    #[test]
    fn test_self_derivation() {
        fn f<T>(x: T) -> T
            where T: From<f64> + Float
        {
            // function val computed here.
            x
        }
        fn df_dx(_: f64) -> f64 {
            // closed form derivative code here.
            1f64
        }
        for val in -5..5 {
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let fval = f(x);
            let dval = df_dx(x);
            let adval = f(ad);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }

    #[test]
    fn test_addition_with_const() {
        fn f<T>(x: T) -> T
            where T: From<f64> + Float
        {
            // function val computed here.
            T::one() + x
        }
        fn df_dx(_: f64) -> f64 {
            // closed form derivative code here.
            1f64
        }
        for val in -5..5 {
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let fval = f(x);
            let dval = df_dx(x);
            let adval = f(ad);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }

    #[test]
    fn test_mul() {
        fn f<T>(x: T) -> T
            where T: From<f64> + Float
        {
            // function val computed here.
            x * x * From::from(2f64)
        }
        fn df_dx(x: f64) -> f64 {
            // closed form derivative code here.
            4f64 * x
        }
        for val in -5..5 {
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let fval = f(x);
            let dval = df_dx(x);
            let adval = f(ad);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }


    #[test]
    fn test_simplest_division() {
        fn f<T>(x: T) -> T
            where T: Float + From<f64>
        {
            T::one() / x
        }
        fn df_dx(x: f64) -> f64 {
            -1f64 / x.powi(2)
        }
        for val in -5..5 {
            if val == 0 {
                continue;
            }
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let adval = f(ad);
            let fval = f(x);
            let dval = df_dx(x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }

    #[test]
    fn test_simplest_division_v2() {
        fn f<T>(x: T) -> T
            where T: Float + From<f64>
        {
            // (T::one() + x) / (T::one() - x)
            T::one() / (T::one() - x)
        }
        fn df_dx(x: f64) -> f64 {
            // 2f64 / (1f64 - x).powi(2)
            1f64 / (1f64 - x).powi(2)
        }
        for val in -5..5 {
            if val == 1 {
                continue;
            }
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let adval = f(ad);
            let fval = f(x);
            let dval = df_dx(x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }

    #[test]
    fn test_powi() {
        fn f<T>(x: T) -> T
            where T: Float + From<f64>
        {
            x.powi(3)
        }
        fn df_dx(x: f64) -> f64 {
            3f64 * x * x
        }
        for val in -5..5 {
            if val == 0 {
                continue;
            }
            let x = val as f64;
            let ad = AutoDiff::var(x);
            let adval = f(ad);
            let fval = f(x);
            let dval = df_dx(x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
        }
    }

    #[test]
    fn test_mul_with_autodiff() {
        fn f<T>(x: T) -> T
            where T: Float + From<f64>
        {
            <T as From<f64>>::from(3f64) * x * x
        }
        fn df_dx(x: f64) -> f64 {
            6f64 * x
        }

        let mut x = -5f64;
        while x <= 5f64 {
            let adval = f(AutoDiff::var(x));
            let fval = f(x);
            let dval = df_dx(x);
            println!("x = {}", x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
            x += 0.5f64;
        }
    }

    #[test]
    fn test_division() {
        fn f<T>(x: T) -> T
            where T: Float + From<f64>
        {
            // f(x) = (1 + x)/(3*x^2)
            // f'(x) = (3*x^2 - 6*(1 + x)*x) / (9*x^4)
            //       = -(x + 2) / (3*x^3)
            (T::one() + x) / (<T as From<f64>>::from(3f64) * x.powi(2))
        }
        fn df_dx(x: f64) -> f64 {
            -(x + 2f64) / (3f64 * x.powi(3))
        }

        let mut x = -5f64;
        while x <= 5f64 {
            if x == 0f64 {
                x += 0.5f64;
                continue;
            }
            let adval = f(AutoDiff::var(x));
            let fval = f(x);
            let dval = df_dx(x);
            println!("x = {}", x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
            x += 0.5f64;
        }
    }

    #[test]
    fn test_simple_expr() {
        fn f<T: Float + From<f64>>(x: T) -> T {
            // (1 + 3*x - 3*x^2 + 5*x^3) / x
            (T::one() + <T as From<f64>>::from(3f64) * x -
             <T as From<f64>>::from(3f64) * x.powi(2) +
             <T as From<f64>>::from(5f64) * x.powi(3)) / x
        }
        fn df_dx(x: f64) -> f64 {
            // (10*x^3 - 3*x^2 - 1) / x^2
            (10f64 * x.powi(3) - 3f64 * x.powi(2) - 1f64) / x.powi(2)
        }
        let mut x = -5f64;
        while x <= 5f64 {
            if x == 0f64 {
                x += 0.5f64;
                continue;
            }
            let adval = f(AutoDiff::var(x));
            let fval = f(x);
            let dval = df_dx(x);
            println!("x = {}", x);
            assert_almost_equal!(adval.val(), fval);
            assert_almost_equal!(adval.dval(), dval);
            x += 0.5f64;
        }
    }
}
