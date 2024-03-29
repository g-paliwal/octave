function [theta, J_history] = gradientDescent_ex(X, y, theta, alpha, num_iters)

%GRADIENTDESCENT Performs gradient descent to learn theta

%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 

%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values


m = length(y); % number of training examples


J_history = zeros(num_iters, 1);



for iter = 1:num_iters

    
% ====================== YOUR CODE HERE ======================
    
% Instructions: Perform a single gradient step on the parameter vector
    
%               theta. 
    
%
    
% Hint: While debugging, it can be useful to print out the values
    
%       of the cost function (computeCost) and gradient here.
    
%







    

Hyp = X * theta;
errVec= Hyp - y;
Grd_chg = alpha / m * (X' * errVec);
theta = theta - Grd_chg ;

% ============================================================

    
% Save the cost J in every iteration    
   
J_history(iter) = 0.5 / m *  sum(((X * theta) - y ) .^2 );%computeCost(X, y, theta);


end



end