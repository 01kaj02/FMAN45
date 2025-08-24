function index = K_means_classifier(x, C)

 N = size(C, 2); 
 dist = zeros(N, 1);

    for i = 1:N
         dist(i) = norm(x - C(:, i));
    end
    [~, index] = min(dist);
end


