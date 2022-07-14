function samples = metro(X_initial,N,M,r)
    % what the inputs mean
        % theta - starting point for the Metropolis-Hastings scheme
        % N - how many Metropolis-Hastings processes you want to do
        % M - how many iterations of the Metropolis-Hastings
        % r - input radii
    L = length(r);
    samples = zeros(M+1,N,L);
    samples(1,:,:) = X_initial;
    for i=2:M+1
        X_new = rando(X_initial,N,r);
        samples(i,:,:) = X_new;
        X_initial = X_new;
    end % runs several iterations of the Metropolis-Hastings scheme
    % keeps the sequence of the values generated stored in each row of
    % samples
end