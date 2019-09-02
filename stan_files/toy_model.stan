functions {
    real hubble_integrand(real z, real xc, real[] theta, real) {
        
      real omegam = theta[1]
      real omegade = theta[2]
      real int_arg;
      real w = -1.0
      int_arg =  (omegam*(1+z)^3 + omegade*(1+z)^(3.+3.*w) + (1.-omegam-omegade)*(1+z)^2.);

      if (!(int_arg>0))
        reject("hubble_integrand: arg must be positive, found arg= ", int_arg);
    
    return 1. / sqrt(int_arg);
        }
    
    vector hubble(real z, real omegam, real omegade) {
        real w = -1.0;
        real xc = 1.0;
        real zp = z;
        function I = hubble_integrand(z, xc, omegam, omegade, w);
        return integrate_1d(I, 0., zp, omegam, omegade, w);
        }
}
data {
    int<lower=0> N;
    vector[N] z;
    vector[N] m;
}
parameters {
    real<lower=0, upper=2> omegam;
    real<lower=0, upper=2> omegade;
    real M_0;
    real<lower=-5, upper=0> log_sigma_int;
}
model {
    omegam ~ uniform(0,2);
    omegade ~ uniform(0,2);
    M0 ~ uniform(-20, -18);
    log_sigma_int ~ uniform(-5, 0)
    mu = hubble(z, omegam, omegade)
    m ~ normal(mu + M_0, np.exp(log_sigma_int));
}

