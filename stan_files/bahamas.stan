functions {
    real hubble_integrand(real z,
                          real xc, 
                          real[] theta,
                          real[] x_r,
                          int[] x_i) {
      real w = -1.0;
      real omegam = theta[1];
      real omegade = theta[2];
      real int_arg;

      int_arg = (omegam*(1+z)^3 + omegade*(1+z)^(3.+3.*w)
                 + (1.-omegam-omegade)*(1+z)^2.);

        if (!(int_arg>0))
            reject("hubble_integrand: arg must be positive, found arg= ", int_arg);

        return 1. / sqrt(int_arg);
    }
    
    vector hubble_int(vector z, real[] theta, real[] x_r, int[] x_i) {

      int nobs = num_elements (z);
      vector[nobs] segments;
      real omegam = theta[1];
      real omegade = theta[2];
      real w = -1.0;
      real leftlim = 0.;
      vector[nobs] hubbleint;
      vector[nobs] integral;

      segments[1] =  integrate_1d(hubble_integrand, 0., z[1], { omegam, omegade }, x_r, x_i, 1.5e-8);

      for (i in 2:nobs){
        segments[i] = integrate_1d(hubble_integrand, z[i-1], z[i], { omegam, omegade }, x_r, x_i, 1.5e-8);
      }
      // now dot the cumulatively summed segments with the respective redshifts
      //for (i in 2:nobs)
      //  hubbleint[i] = segments[i-1] + segments[i];
      hubbleint = cumulative_sum(segments);
      integral = hubbleint .* z;

      return integral;
    }
    
    vector Dlz(vector z, real[] theta, real[] x_r, int[] x_i) {

      int nobs = num_elements(z);
      real omegam = theta[1];
      real omegade = theta[2];
      vector[nobs] hubbleint; 
      real h = 0.72;
      real omegakmag = fabs(1-omegam-omegade);
      real c_light =  299792.0;
      vector[nobs] z_h = z;
      vector[nobs] dist;
      
      hubbleint = hubble_int(z, { omegam, omegade }, x_r, x_i);

      
      if ((omegade + omegam) > 1)
        for (i in 1:nobs)
          dist[i] = ((c_light * 10e-5) * (1 + z_h[i]) / (h * omegakmag)) * sin(hubbleint[i] * omegakmag);

      else if ((omegade + omegam == 1))
        for (i in 1:nobs)
          dist[i] = ((c_light * 10e-5) * (1 + z_h[i]) * hubbleint[i] / h);

      else
        for (i in 1:nobs)
          dist[i] = ((c_light * 10e-5) * (1 + z_h[i]) / (h * omegakmag)) * sinh(hubbleint[i] * omegakmag);
        
      return dist;
    }

    vector muz(vector z, real[] theta, real[] x_r, int[] x_i) {
   
      real omegam = theta[1];
      real omegade = theta[2];

      return 5.0 * log10(Dlz(z, { omegam, omegade }, x_r, x_i)) + 25.;
    }

}


data {
  int<lower=0> N;
  vector[N] z;
  vector[N] m;
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0, upper=2> omegam;
  real<lower=0, upper=2> omegade;
  real<lower=-20, upper=-18> M_0;
  real<lower=-5, upper=0> log_sigma_int;
 }
transformed parameters {
  real sigma_int = exp(log_sigma_int);
  vector[N] mu;
  // mu[1] = muz({ omegam, omegade }, z, x_i)
  // for (i in 1:N)
  mu = muz(z, { omegam, omegade }, x_r, x_i);
  for (i in 1:N){
    print("i: ", i, " mu[i] ", mu[i]);
  }
}
model {
  omegam ~ uniform(0,2);
  omegade ~ uniform(0,2);
  M_0 ~ uniform(-20, -18);
  log_sigma_int ~ uniform(-5, 0);
  
  for (i in 1:N){
    m[i] ~ normal(mu[i] + M_0, sigma_int);
  }
}

