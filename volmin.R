install.packages('vrnmf')
library(vrnmf)
ls("package:vrnmf")

# Set the seed for reproducibility
set.seed(0)

B <- matrix(c(0.04494787, 0.95505213,
              0.4760971, 0.5239029,
              0.10196587, 0.89803413,
              0.79131988, 0.20868012), 
            nrow = 4, ncol = 2, byrow = TRUE)

print("Matrix B:")
print(B)

# Prepare other parameters
vol <- list(U = matrix(runif(8), nrow = 4), eigens = diag(runif(2)))
n.comp <- 2
wvol <- 0.0001
n.iter <- 200000

# Run the function
result <- volnmf_main(vol, B, n.comp = n.comp, wvol = wvol, n.iter = n.iter)

# Print the results
print(result)
C <- matrix(c(0.00000000, 0.4262202,
              0.34936988, 0.1799099,
              0.04497373, 0.3938700,
              0.60565639, 0.0000000), 
            nrow = 4, ncol = 2, byrow = TRUE)
R <- matrix(c(1.3484467, 0.06460641,
              0.5740275, 2.16745600), 
            nrow = 2, ncol = 2, byrow = TRUE)
print(C%*%R)
