##################################################
# Analysis of PSA recurrence in prostate cancer patients
# Author: Lukas Owens
# Date written: 6/10/2025
##################################################

##################################################
# Set up
##################################################
library(tidyverse)
library(nlme)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

##################################################
# Load and prepare data in required format
##################################################
load('data/msk_data_processed.RData')
dset <- all_data$patients |> filter(include_flag==1)
dset <- dset |> head(n=500)
pset <- all_data$psa |> filter(include_flag==1)
cset <- inner_join(dset, pset, by='ptid')
cset <- cset |> filter(is.na(tt_salvage_tx) | psa_time < tt_salvage_tx)
cset <- cset |> filter(psa_time>0.5)
cset <- cset |> mutate(tx_grade_group=fct_recode(tx_grade_group, '2-6'='<=6'))
cset <- cset |> select(ptid, tx_grade_group, tx_tstage, psa_time, sms, lni, logpsa, tx_year)
cset <- cset |> nest(data=c(psa_time, logpsa))
cset <- cset |> mutate(id=row_number())
cset <- cset |> unnest(data)

##################################################
# Fit LMM with nlme::lme()
##################################################
fit_0 <- lme(logpsa ~ psa_time + 
                 psa_time:tx_grade_group + 
                 psa_time:tx_tstage + 
                 psa_time:lni + 
                 psa_time:sms,
             random=reStruct(~ 1 + psa_time | ptid, pdClass='pdDiag', REML=FALSE),
             data=cset,
             control = nlme::lmeControl(opt = 'optim'),
             method = 'ML')
save(fit_0, file='output/fit_0.RData')

##################################################
# Fit models using Stan
##################################################
fe_formula <- ~ 1 + 
    psa_time + 
    psa_time:tx_grade_group + 
    psa_time:tx_tstage + 
    psa_time:lni + 
    psa_time:sms
x <- model.matrix(fe_formula, data=cset)
model_data <- list(Nobs=nrow(cset),
                   Npreds=ncol(x),
                   Ngroups=length(unique(cset$id)),
                   y=cset$logpsa,
                   x=x,
                   time=cset$psa_time,
                   id=as.integer(cset$id))

fit_1 <- stan(file='model-1.stan',
            data=model_data)
save(fit_1, file='output/stan_out_1.RData')

fit_2 <- stan(file='model-2.stan',
            data=model_data)
save(fit_2, file='output/stan_out_2.RData')
