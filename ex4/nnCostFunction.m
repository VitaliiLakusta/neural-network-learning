function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

m = size(X, 1);
X = [ones(m, 1) X];

% Feedforward propagation
% Theta1 % h x n
% Theta2 % r x h

z2 = X * Theta1'; % m x h
a2 = sigmoid(z2); % m x h
a2WithBias = [ones(size(a2, 1), 1) a2]; % m x (h + 1)

z3 = a2WithBias * Theta2'; % m x r
a3 = sigmoid(z3); % m x r

% Computing cost function
for i=1:m
    yVector = zeros(num_labels, 1);
    yVector(y(i)) = 1;
    hypothesisVector = a3(i, :);
    hypothesisVector = hypothesisVector';
    J += sum( ...
        -yVector .* log(hypothesisVector) - (1 - yVector) .* log(1 - hypothesisVector) ...
    );
end
J = (1.0 / m) * J;

% Add regularization term to cost function
theta1SumSquared = sum(sum(Theta1(:, 2:end) .^ 2));
theta2SumSquared = sum(sum(Theta2(:, 2:end) .^ 2));
J += (lambda / (2 * m)) * (theta1SumSquared + theta2SumSquared);

% Backpropagation
yMatrix = zeros(m, num_labels); % m x r
for i=1:m
    yMatrix(i, y(i)) = 1;
end

delta3 = a3 .- yMatrix; % m x r
delta2 = delta3 * Theta2(:, 2:end) .* sigmoidGradient(z2);   % [m x r] x [r x h] --> [m x h]

D1 = delta2' * X;  % [h x m] x [m x n] --> [h x n]
D2 = delta3' * a2WithBias; % [r x m] x [m x (h+1)] --> [r x (h+1)]

Theta1_grad = (1.0 / m) .* D1;
Theta2_grad = (1.0 / m) .* D2;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
