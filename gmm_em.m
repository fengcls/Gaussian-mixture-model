function phi_max = gmm_em(X,K)
%% EM-GMM
disp('===== Expectation Maximization Gaussian Mixture Model =====')
Niter = 100;
if size(X,1)>size(X,2)
    X = X';
end;
N   = size(X,2);
d   = size(X,1);

h1 = figure;
if d<=3;h2 = figure;end;

for kK = 1:length(K)
    clearvars -except Niter N d X K kK phi_max h1 h2;
    % initialization
    pi_d(:,1) = 1/K(kK)*ones(1,K(kK));
    rng(1,'twister')
    mu(:,:,1) = mean(X,2)*ones(1,K(kK))+randn(d,K(kK));
    Sigma_cov  = zeros(d,d,K(kK),Niter);
    phi = zeros(N,K(kK),Niter);
    n = zeros(K(kK),Niter);
    ft = zeros(Niter,1);
    
    for kj = 1:K(kK)
        Sigma_cov(:,:,kj,1) = cov(X');
    end;
    
    fttmp = zeros(N,1);
    for kj = 1:K(kK)
        fttmp = fttmp + pi_d(kj,1)*mvnpdf(X',mu(:,kj,1)',Sigma_cov(:,:,kj,1));
    end;
    ft(1) = sum(log(fttmp));
    
    for kiter = 2:Niter
        % E-step
        for kj = 1:K(kK)
            % NxKxkiter
            phi(:,kj,kiter) = pi_d(kj,kiter-1)*...
                mvnpdf(X',mu(:,kj,kiter-1)',Sigma_cov(:,:,kj,kiter-1));
        end;
        phi(:,:,kiter) = phi(:,:,kiter)./(sum(phi(:,:,kiter),2)*ones(1,K(kK)));
        % M-step
        n(:,kiter) = squeeze(sum(phi(:,:,kiter)));
        pi_d(:,kiter)  = n(:,kiter)/N;
        for kj = 1:K(kK)
            mu(:,kj,kiter) = sum(ones(d,1)*phi(:,kj,kiter)'.*X,2)/n(kj,kiter);
            Sigma_cov(:,:,kj,kiter) = ...
                ((X-mu(:,kj,kiter)*ones(1,N)).*(ones(d,1)*sqrt(phi(:,kj,kiter))'))*...
                ((X'-ones(N,1)*mu(:,kj,kiter)').*(sqrt(phi(:,kj,kiter))*ones(1,d)))/n(kj,kiter);
        end;
        
        fttmp = zeros(N,1);
        for kj = 1:K(kK)
            fttmp = fttmp + pi_d(kj,kiter)*mvnpdf(X',mu(:,kj,kiter)',Sigma_cov(:,:,kj,kiter));
        end;
        ft(kiter) = sum(log(fttmp));
    end;
    figure(h1);
    subplot(length(K),1,kK)
    plot(ft)
    xlabel('Iteration')
    ylabel('Log Likelihoo')
    title(['EM-GMM K=',num2str(K(kK))])
    ylim([min(ft)-range(ft)/100,max(ft)+range(ft)/100])
    
    if d<=3
        figure(h2);
        subplot(length(K),1,kK)
        [~,phi_max(:,kK)] = max(phi(:,:,Niter),[],2);
        for ksc = 1:K(kK)
            hold on;
            if d==2
                scatter(X(1,phi_max(:,kK)==ksc),X(2,phi_max(:,kK)==ksc))
            elseif d==3
                scatter3(X(1,phi_max(:,kK)==ksc),X(2,phi_max(:,kK)==ksc),X(3,phi_max(:,kK)==ksc))
            end;
        end;
    end;
    title(['EM-GMM K=',num2str(K(kK))])
end;

