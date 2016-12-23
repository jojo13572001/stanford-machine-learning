function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
theta_num = length(theta);
temp_derivative = zeros(size(theta));
for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sigma = X*theta-y;
    for theta_iter = 1:theta_num
    %for sample_iter = 1:m
            %temp = temp + (X(sample_iter,:)*theta-y(sample_iter))*X(sample_iter,theta_iter);
        temp_derivative(theta_iter) = sum(sigma.*X(:,theta_iter));
    end
    
    theta = theta - alpha/m*temp_derivative;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X,y,theta);

end
%disp(J_history);
end
