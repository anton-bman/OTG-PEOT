function lambda_final = newton_armijo(A, K, r, epsilon, gamma, lambda0, tol, maxIter, onesN) 
% Solve for lambda using Newton's method with Armijo line search

lambda = lambda0;
c = 1e-1;

for iter = 1:maxIter
    z = A' * lambda / epsilon;
    u = exp(z);
    
    K1 = K * onesN;
    f = (r - A*(K1.*u) - (1/(2*gamma))*lambda);
    
    %%%% FAST CODE %%%%
    Hfun_posdef = @(x) ( A * ( (u .* K1) .* (A' * x) ) / epsilon + (1/(2*gamma)) * x );

    % Solve: H_posdef * Delta_lambda = f
    tol = 1e-6;
    maxit = 100;

    [Delta_lambda, ~, ~] = pcg(Hfun_posdef, f, tol, maxit);
    
    % Line search
    alpha = 2;
    counter = 0;
    while true
        lambda_new = lambda + alpha * Delta_lambda;
        z_new = A' * lambda_new / epsilon;
        u_new = exp(z_new);
        
        g_new = r.'*lambda_new-epsilon*u_new.'*K1 - 1/(4*gamma)*norm(lambda_new)^2;
        g =     r.'*lambda-epsilon*u.'*K1     - 1/(4*gamma)*norm(lambda)^2;

        counter = counter +1;
    
        %%% Stopping condition for our problem
        if g_new >= g + c*alpha*f.'*Delta_lambda
            break;
        else
            alpha = alpha /2;
        end
        if alpha < 1e-12
            warning('Step size too small');
            break;
        end
        
    end
    
    lambda = lambda_new;
    
    % convergence check
    if norm(Delta_lambda) < 1e-7 
        break;
    end
    
end

lambda_final = lambda;

end
