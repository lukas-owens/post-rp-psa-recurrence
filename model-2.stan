data {
    int<lower=0> Nobs;
    int<lower=0> Npreds;
    int<lower=0> Ngroups;
    
    vector[Nobs] y;
    matrix[Nobs,Npreds] x;
    vector[Nobs] time;
    
    int<lower=1,upper=Ngroups> id[Nobs];
}

parameters {
    vector<lower=0>[Npreds] beta; // Fixed effects coefficients
    real<lower=0> sigma; // Std dev of noise
    
    vector<lower=0>[Ngroups] b_0_raw; // Unscaled random intercepts
    real<lower=0> tau_0;     // Std dev of random intercepts
    
    vector<lower=0>[Ngroups] b_1_raw; // Unscaled random slopes
    real<lower=0> tau_1;     // Std dev of random slopes
    
}

transformed parameters {
    vector[Ngroups] b_0 = b_0_raw * tau_0;
    vector[Ngroups] b_1 = b_1_raw * tau_1;
    vector[Nobs] fixed_effects = x*beta;
}

model {
    // Priors
    tau_0 ~ exponential(1);
    b_0_raw ~ exponential(1);
    tau_1 ~ exponential(1);
    b_1_raw ~ exponential(1);
    sigma ~ exponential(1);
    
    // Likelihood
    for (n in 1:Nobs)
        y[n] ~ normal(fixed_effects[n] + b_0[id[n]] + b_1[id[n]]*time[n] , sigma);
}
