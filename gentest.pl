use strict;
use warnings;
use feature qw(say);

use Getopt::Long;
use Pod::Usage;

my %options;
GetOptions(\%options,
    "skip-zero-in-range",
    "help",
    "range=s",
) or pod2usage();

pod2usage(-perldoc => 1) if $options{help};

my $name  = shift @ARGV or pod2usage();
my $range = $options{range} || "-5..5";

my $skip_zero_code = $options{"skip-zero-in-range"} ?
    "if val == 0 { continue; }" : "";

print <<EOT

#[test]
fn ${name}() {
    fn f<T>(x: T) -> T where T: From<f64> + Float {
        // function val computed here.
        T::zero()
    }

    fn df_dx(x: f64) -> f64 {
        // closed form derivative code here.
        0f64
    }

    for val in $range {
        $skip_zero_code
        let x = val as f64;
        let ad = AutoDiff::var(x);
        let fval = f(x);
        let dval = df_dx(x);
        let adval = f(ad);

        assert_almost_equal!(adval.val(), fval);
        assert_almost_equal!(adval.dval(), dval);
    }
}

EOT

__END__

=pod

=head1 NAME

gentest.pl

=head1 SYNOPSIS

    perl gentest.pl [OPTIONS] NAME [RANGE]

For example,

    perl gentest.pl test_division --range=-5..5

will generate:

    #[test]
    fn test_division_more() {
        fn f<T>(x: T) -> T where T: From<f64> + Float {
            // function val computed here.
            T::zero()
        }

        fn df_dx(x: f64) -> f64 {
            // closed form derivative code here.
            0f64
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

=head1 DESCRIPTION

Generate test function skeleton for autodiff.

=cut
