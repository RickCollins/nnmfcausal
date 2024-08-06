install.packages('vrnmf')
library(vrnmf)
ls("package:vrnmf")

# Set the seed for reproducibility
set.seed(0)

B <- matrix(c(0.11884672, 0.88115328,
              0.88045151, 0.11954849,
              0.72660589, 0.27339411,
              0.26953994, 0.73046006), 
            nrow = 4, ncol = 2, byrow = TRUE)

print("Matrix B:")
print(B)

# Prepare other parameters
vol <- list(U = matrix(runif(8), nrow = 4), eigens = diag(runif(2)))
n.comp <- 2
wvol <- 0.01
n.iter <- 1000000

# Run the function
result <- volnmf_main(vol, B, n.comp = n.comp, wvol = wvol, n.iter = n.iter)

# Print the results
print(result)

R <- result$R
C <- result$C

print(C%*%R)
print(B)
