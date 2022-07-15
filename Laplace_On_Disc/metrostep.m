function phi_out = metrostep(r,theta,K,phi_init)
    % r is an array of the input radii
    % theta is an array of the input angles
    % K is the batch size
    M = length(r);
    N = length(theta);
    
    % creates a mesh of points (r,theta) with the number of batches desired
    R = repmat(r,[N 1 K]);
    Theta = repmat(theta',[1 M K]);
    
    % generates a random proposal phi in the range [-pi,pi]
    % under the current setup, we need to input phi_init, since there needs
    % to be a way to keep track of things in the metro algorithm
    % phi_init = -pi + 2*pi*rand(N,M,K);
    phi_prop = -pi + 2*pi*rand(N,M,K);
    
    % computes the proposal distribution ratio f(phi_prop)/f(phi_init)
    dist_ratio = (1-2*R.*cos(Theta-phi_init)+R.^2)./(1-2*R.*cos(Theta-phi_prop)+R.^2);
    
    % computes the acceptance probability
    A = min(ones(N,M,K),dist_ratio);
    
    % generates variables from U[0,1] and determines whether to accept or
    % reject phi_prop
    U = rand(N,M,K);
    acc_or_rej = U <= A;
    
    % updates initial guess for phi, which will be the output of this
    % entire function
    phi_out = phi_init;
    phi_out(find(acc_or_rej==1)) = phi_prop(find(acc_or_rej==1));
end