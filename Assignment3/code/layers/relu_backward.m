function dldX = relu_backward(X, dldZ)
    %error('Implement this!');
    dldX = dldZ .* (X > 0);
end
