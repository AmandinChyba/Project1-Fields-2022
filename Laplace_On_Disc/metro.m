function [values_to_plot,error] = metro(r,theta,K,T,phi_init)
    % the inputs here should be
        % r = array of input radii
        % theta = array of input angles
        % K = batch size
        % T = time steps for the Metropolis-Hastings
    tic;
    M = length(r);
    N = length(theta);
    R = repmat(r,[N 1 K]);
    Theta = repmat(theta',[1 M K]);
    phi_samples = zeros(T,N,M,K);
    % mean = zeros(N,M);
    g_int = zeros(T,N,M,K);
    for i=1:T
        phi_out = metrostep(R,Theta,K,phi_init,N,M);
        phi_samples(i,:,:,:) = phi_out;
        phi_init = phi_out;
    end % runs several iterations of the Metropolis-Hastings scheme
    % keeps the sequence of the values generated stored in each row of
    % samples
    % at this point we generate the average of the samples
    % g = sin(phi_samples(100:end,:,:,:));
    % g = exp(sin(phi_samples(T/10:end,:,:,:))); % special g that Adam gave
    g = g_int(100:end,:,:,:); 
    g(phi_samples(100:end,:,:,:) > 0) = 1; % calculating g in the 2019 paper
    values_to_plot = reshape(mean(g,[1 4]),[N M]);
    % for plotting the results - 
    R_for_plot = repmat(r,[N 1]);
    Theta_for_plot = repmat(theta',[1 M]);
    [x,y,z] = pol2cart(Theta_for_plot,R_for_plot,values_to_plot); % converts to cartesian so that we can plot it in 3d 
    surf(x,y,z); % surface plot
    
    % to find the errors, perform the following steps
    % R = repmat(r,[N 1]);
    % Theta = repmat(theta',[1 M]);
    % Exact = R.*sin(Theta); % this can be used if we know what the solution to Laplace's equation is
%   % when we don't my guess is that we just do a finer grid approximation
    Exact = 0.5 + (1/pi).*atan((2.*y)./(1-x.^2-y.^2)); % this is for the arctan function solution
    error = values_to_plot - Exact;
%    max_error = max(error,[],'all');
%    mean_error = mean(error,'all');
    toc;
end