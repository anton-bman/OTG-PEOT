function [M, lambda] = transport_newton(r_real, C_small,lambda, M, A_real, gamma, maxIter)
tol = 1e-3;
epsilon = 1e-2;

prev_transport_cost = inf;
%%%%%%%%  Optimize over M, omega
%%% Step 1: initialize C_small (prior of pitch) -- input to function
%%% Step 2: iterate (k = 1, ..., K)   
for k = 1:maxIter

    if k == 1
        old_M = ones(size(C_small));
    else
        old_M = M;
    end

    %%% Step 3: update C^(k)
    C_hat = C_small - epsilon*log(old_M);
    %form K
    K = exp(-C_hat/epsilon);
    counter = 0;
    rel_change = 0;



    %%% step 4: while Newton not converged do
    while true
        counter = counter + 1;
        lambda_new = newton_armijo(A_real, K, r_real, epsilon, gamma, lambda, tol, maxIter, ones(size(C_hat,2),1));

        rel_change(counter) = norm(lambda_new - lambda) / norm(lambda_new);

        logM = (-C_hat + (A_real'*lambda_new)*ones(1, size(C_hat,2)) )/epsilon;
        C_hat = C_small - epsilon*logM;
        K = exp(-C_hat/epsilon);
        if counter > 1 && rel_change(counter) < 1e-2
            lambda = lambda_new;
            break;
        end
        lambda = lambda_new;
    end

    %%%% step 5: calculate M
    M = exp(logM);

    


    deltaval = -lambda/(2*gamma);
    transport_cost(k) = sum(sum(C_small.*M)) + gamma*sum(deltaval.^2);

    errorVal = abs(transport_cost(k) - prev_transport_cost)/norm(transport_cost(k));
    if errorVal <= tol
        break;
    end

    prev_transport_cost = transport_cost(k);

end



end