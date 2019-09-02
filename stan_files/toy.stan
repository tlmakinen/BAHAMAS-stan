
functions {
    real normal_density(real x,          // Function argument
                    real xc,         // Complement of function argument
                                     //  on the domain (defined later)
                    real[] theta,    // parameters
                    real[] x_r,      // data (real)
                    int[] x_i) {     // data (integer)

    real mu = theta[1];
    real sigma = theta[2];

      return 1 / (sqrt(2 * pi()) * sigma) * exp(-0.5 * ((x - mu) / sigma)^2);
    }

    real toy_hubble(real left_limit, real[] theta, real[] x_r, int[] x_i) {
      real mu = theta[1];
      real sigma = theta[2];
          
      return integrate_1d(normal_density, left_limit, positive_infinity(), { mu, sigma }, x_r, x_i, 1e-8);
    }

}


data {
  int N;
  real y[N];
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real mu;
  real<lower = 0.0> sigma;
  real left_limit;
}

model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1);
  left_limit ~ normal(0, 1);
  target += normal_lpdf(y | mu, sigma);
  target += log(toy_hubble(left_limit, { mu, sigma }, x_r, x_i));

}
