# Reproduce every number, table and figure in the README.
# Run from the repository root:  Rscript run_all.R   (about 1 minute)
source("R/01_explore.R")
source("R/02_models.R")
source("R/03_costs.R")
source("R/04_interpret.R")
cat("\nDone. Results in output/, figures in figures/.\n")
