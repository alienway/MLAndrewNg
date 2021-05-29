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

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
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
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1), X];
a1 = X; %m*(input_layer_size +1)
z2 = a1*Theta1'; % m*hidden_layer_size
a2 = sigmoid(z2); % m*hidden_layer_size
a2 = [ones(m,1), a2]; % m*(hidden_layer_size+1)
z3 = a2*Theta2'; %m*num_labels
a3 = sigmoid(z3); %m*num_labels, in probabilities for each labels

%wrong. check formula. no need. [maxProb, a3] = max(a3, [], 2); %m*1
%wrong. check formula. no need. a3 = (a3 == [1:num_labels]); %m*num_labels, in zeros and ones

%convert y from m*1 to m*num_labels
y = (y == [1:num_labels]); %m*num_labels

%sum across each row ie sum over num_labels, then take mean over m ie mean by col
%this implementation is confusing to debug
%J = sum(mean(-y.*log(a3) - (1-y).*log(1-a3)));

%Alternative, sum across row and then take mean
J = mean(sum(-y.*log(a3) - (1-y).*log(1-a3),2));

%the below won't work because we need element wise multiplication
%J = sum(-y'*log(a3) - (1-y)'*log(1-a3))/m;

%Alternative, specifically divide by m
%J = sum(sum(-y.*log(a3) - (1-y).*log(1-a3)))/m;

%below for regularised J
Theta1Reg = Theta1(:, 2:end);
Theta2Reg = Theta2(:, 2:end);

J = mean(sum(-y.*log(a3) - (1-y).*log(1-a3),2)) ...
  +lambda/(2*m)*(sum(sum(Theta1Reg.*Theta1Reg)) + sum(sum(Theta2Reg.*Theta2Reg)));


%below for backward probagation
d3 = a3 - y; %m*num_labels
d2 = d3*Theta2Reg .*sigmoidGradient(z2); 
%Theta2Reg: num_labels*hidden_layer_size
%z2 and d2: m*hidden_layer_size. 
Theta2_grad = (d3'*a2)/m;
%Theta2_grad: num_labels*(hidden_layer_size +1)
%a2: m*(hidden_layer_size+1)
Theta1_grad = (d2'*a1)/m;
%a1: m*(input_layer_size+1)
%Theta1_grad: hidden_layer_size*(input_layer_size +1)

%below adds regularization terms
Theta2_grad = Theta2_grad + lambda/m*[zeros(num_labels, 1), Theta2Reg];
Theta1_grad = Theta1_grad + lambda/m*[zeros(hidden_layer_size, 1), Theta1Reg];








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
