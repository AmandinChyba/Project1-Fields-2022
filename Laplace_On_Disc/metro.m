function [phi_samples,values_to_plot] = metro(r,theta,K,T,phi_init)
    % the inputs here should be
        % r = array of input radii
        % theta = array of input angles
        % K = batch size
        % T = time steps for the Metropolis-Hastings
    M = length(r);
    N = length(theta);
    phi_samples = zeros(T,N,M,K);
    for i=1:T
        phi_out = metrostep(r,theta,K,phi_init);
        phi_samples(i,:,:,:) = phi_out;
        phi_init = phi_out;
    end % runs several iterations of the Metropolis-Hastings scheme
    % keeps the sequence of the values generated stored in each row of
    % samples
    % at this point we generate the average of the samples
    g = sin(phi_samples(T/10:end,:,:,:));
    values_to_plot = reshape(mean(g,[1 4]),[N M]);
    
    % to find the errors, perform the following steps
%     R = repmat(r,[N 1]);
%     Theta = repmat(theta',[1 M]);
%     Exact = R.*sin(Theta); % this can be used if we know what the solution to Laplace's equation is
%     % when we don't my guess is that we just do a finer grid approximation
%     error = abs(values_to_plot - Exact);
%     disp('The maximum error is ' + max(error,[],'all'));
%     disp('The average error is ' + mean(error,'all'));
end