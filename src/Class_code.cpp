#include <Rcpp.h>
using namespace Rcpp;

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

//' Stochastic Gradient Descent for Least Squares with Ridge Regularization
 //'
 //' @param y Response vector (n x 1)
 //' @param A Predictor matrix (n x p)
 //' @param x0 Initial coefficients (p x 1)
 //' @param lambda Regularization parameter
 //' @param batch Batch size (1 to n)
 //' @param initial_step_size Initial step size for gradient descent
 //' @param tol Convergence tolerance
 //' @param max_iter Maximum number of iterations
 //' @param printing Whether to print the convergence message
 //' @return A list containing coefficients, differences, and losses
 //' @export
 // [[Rcpp::export]]
 Rcpp::List stochastic_gradient_descent_lsq(const arma::vec& y,
                                              const arma::mat& A,
                                              arma::vec x0,
                                              double lambda,
                                              int batch,
                                              double initial_step_size = 1,
                                              double tol = 1E-6,
                                              int max_iter = 10000,
                                              bool printing = false) {
   int n = A.n_rows;
   int p = A.n_cols;
   
   // Sanity checks
   if (n != y.n_elem) {
     Rcpp::stop("Length of y must match the number of rows in A");
   }
   if (n < p) {
     Rcpp::stop("Number of observations (n) must be greater than or equal to the number of predictors (p)");
   }
   
   arma::vec x = x0;
   arma::vec prev_loss(1, arma::fill::zeros);
   arma::vec loss_rec(max_iter, arma::fill::zeros);
   arma::vec diff_rec(max_iter, arma::fill::zeros);
   double diff = std::numeric_limits<double>::infinity();
   
   int iter = 0;
   
   while (iter < max_iter && diff > tol) {
     // Sample batch
     arma::uvec indices = arma::randperm(n, batch);
     arma::mat Asub = A.rows(indices);
     arma::vec ysub = y.elem(indices);
     
     // Calculate gradient
     arma::mat AA = Asub.t() * Asub;
     arma::vec Ay = Asub.t() * ysub;
     arma::vec grad = (AA * x - Ay) / batch + 2 * lambda * x / n;
     
     // Update coefficients
     x = x - (initial_step_size / (iter + 1)) * grad;
     
     // Calculate loss
     double loss = 0.5 * arma::dot(y - A * x, y - A * x) + lambda * arma::dot(x, x);
     loss_rec[iter] = loss;
     
     // Calculate difference
     if (iter > 0) {
       diff_rec[iter] = (prev_loss[0] - loss) / std::abs(prev_loss[0]);
       diff = std::abs(diff_rec[iter]);
     }
     
     prev_loss[0] = loss;
     iter++;
   }
   
   if (printing) {
     Rcpp::Rcout << "Converged after " << iter << " steps" << std::endl;
   }
   
   // Return results
   return Rcpp::List::create(Rcpp::Named("x") = x,
                             Rcpp::Named("diff") = diff_rec.subvec(0, iter - 1),
                             Rcpp::Named("loss") = loss_rec.subvec(0, iter - 1));
 }
 

#include <RcppArmadillo.h>
 // [[Rcpp::depends(RcppArmadillo)]]
 
 //' Ridge regression Loss Function for Linear Models
 //'
 //' @param y      A (n x 1) vector of response variables
 //' @param A      A (n x p) matrix of predictor variables
 //' @param x      A (p x 1) vector of effect size for each predictor variable
 //' @param lambda Regularization parameter in ridge regression
 //' @return Ridge Regression errors with penalty
 //' @export
 // [[Rcpp::export]]
 double loss_ridge(const arma::vec& y, const arma::mat& A, const arma::vec& x, double lambda) {
   int n = y.n_elem;
   int p = x.n_elem;
   
   // Sanity check
   if (A.n_rows != n || A.n_cols != p) {
     Rcpp::stop("Dimensions of A must match length of y and x");
   }
   
   // Calculate residuals
   arma::vec res = y - A * x;
   
   // Calculate objective function
   double loss = 0.5 * arma::dot(res, res) + lambda * arma::dot(x, x);
   
   return loss;
 }
 