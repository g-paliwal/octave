function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

temX = [ones(m ,1) X];
hx = X * theta;

mid_sqr_sum =  sum ((hx - y) .^ 2 ) ;
dif_lam = lambda * sum(theta([2:length(theta)],1) .^ 2);
J = 0.5 * (mid_sqr_sum + dif_lam) /m;


mid_grad = sum(X' * (hx-y),2) /m;
size(mid_grad);
grad_lam = lambda * theta / m;
grad_lam(1, 1) = 0;
grad = mid_grad+ grad_lam;











% =========================================================================

grad = grad(:);

end
