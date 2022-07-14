function X_new = rando(X_initial,N,r) 
    
    L = length(r);

    % X_initial = repmat(theta,[1 N L]); 
    
    % N = length(X_initial);
    
    % X = normrnd(0,1)*ones(1,N); % generates random variables from a Gaussian distribution
    
    X = -pi + 2*pi*rand(1,N,L); % generates an array 
    % of N numbers uniformly distributed between -pi and pi
    
    % makes appropriate matrix for the radii
    R = reshape(repmat(r,[N 1]),[1 N L]);
    % computes the proposal condition f(X')/f(X)
    
    % test for Gaussian distribution
    % proposal = exp(0.5.*((X_initial).^2 - X.^2));
    
    % below is the proposal for f being the Poisson kernel
    proposal = (1 - 2*R.*cos(X_initial) + R.^2)./(1 - 2*R.*cos(X) + R.^2);
    
    % goes into calculating the acceptance probability
    a = [ones(1,N,L);proposal];
    
    % calculates the acceptance probability
    A = min(a,[],1);
    
    % creates random values from U[0,1] for the acceptance condition
    U = rand(1,N,L);
    
    % finds whether U <= A
    acc_or_rej = U <= A;
    
    X_new = X_initial;
    % finds which entries of acc_or_rej satisfy acceptance condition (the
    % entries that are equal to 1)
    X_new(find(acc_or_rej==1)) = X(find(acc_or_rej==1)); 
    % apparently there is something wrong with doing the find stuff here,
    % it's a little slow, but this works!
    
    % disp(X_initial);
    % disp(X_new);
end