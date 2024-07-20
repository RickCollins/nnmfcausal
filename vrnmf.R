# Load necessary library
library(vrnmf)

# Load the matrix from CSV
X <- as.matrix(read.csv('matrix_X.csv', header = FALSE))

# Perform the factorization
volnmf_result <- volnmf_main(vol = NULL, B = X, n.comp = 5, wvol = 10.0, n.iter = 10000, err.cut = 1e-4)

# Extract factorized matrices
C <- volnmf_result$C
R <- volnmf_result$R

# Save the factorized matrices to CSV files
write.csv(C, 'matrix_C.csv', row.names = FALSE)
write.csv(R, 'matrix_R.csv', row.names = FALSE)
