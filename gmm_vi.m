function phi_max = gmm_vi(X,K)
%% VI-GMM
disp('===== Variational Inference Gaussian Mixture Model =====')
if size(X,1)>size(X,2)
    X = X';
end;
N   = size(X,2);
d   = size(X,1);
Niter = 100;

h1 = figure;
if d<=3;h2 = figure;end;

for kK = 1:length(K)
    clearvars -except N d Niter X K kK h1 h2;
    % Dirichlet
    alpha = ones(K(kK),1);
    
    rng(1,'twister')
    mu = mean(X,2)*ones(1,K(kK))+randn(d,K(kK));
    % normal
    c = 10;
    % wishart
    a0 = d;
    a = a0*ones(K(kK),1);
    
    A0 = cov(X');
    % B0 = d*A0;
    B0 = d/2*A0;
    logdetB0 = 2*sum(log(diag(chol(B0))));
    
    for kj = 1:K(kK)
        A(:,:,kj,1) = A0;
        B(:,:,kj,1) = B0;
    end;
    for kiter = 2:Niter
        %% a
        for kj = 1:K(kK)
            t1(kj,kiter) = sum(psi((1-(1:d)+a(kj,kiter-1))/2)) - 2*sum(log(diag(chol(B(:,:,kj,kiter-1)))));
            t2(:,kj,kiter) = diag((X-mu(:,kj,kiter-1)*ones(1,N))'*(a(kj,kiter-1)*inv(B(:,:,kj,kiter-1)))*(X-mu(:,kj,kiter-1)*ones(1,N)));
            t3(kj,kiter) = trace(a(kj,kiter-1)*inv(B(:,:,kj,kiter-1))*A(:,:,kj,kiter-1));
            t4(kj,kiter) = psi(alpha(kj,kiter-1))-psi(sum(alpha(:,kiter-1)));
            phi(:,kj,kiter) = exp(0.5*t1(kj,kiter)-0.5*t2(:,kj,kiter)-0.5*t3(kj,kiter)+t4(kj,kiter));
        end;
        phi(:,:,kiter) = phi(:,:,kiter)./(sum(phi(:,:,kiter),2)*ones(1,K(kK)));
        
        %% b
        n(:,kiter) = squeeze(sum(phi(:,:,kiter)));
        
        %% c
        alpha(:,kiter) = alpha(:,1)+n(:,kiter);
        
        %% d
        for kj = 1:K(kK)
            A(:,:,kj,kiter) = inv(1/c*eye(d)+n(kj,kiter)*a(kj,kiter-1)*inv(B(:,:,kj,kiter-1)));
            mu(:,kj,kiter) = A(:,:,kj,kiter)*(a(kj,kiter-1)*inv(B(:,:,kj,kiter-1))*sum(ones(d,1)*phi(:,kj,kiter)'.*X,2));
        end;
        
        %% e
        a(:,kiter) = a(:,1)+n(:,kiter);
        for kj = 1:K(kK)
            B(:,:,kj,kiter) = B(:,:,kj,1) + ((X-mu(:,kj,kiter)*ones(1,N)).*(ones(d,1)*sqrt(phi(:,kj,kiter))'))*...
                ((X'-ones(N,1)*mu(:,kj,kiter)').*(sqrt(phi(:,kj,kiter))*ones(1,d))) + sum(phi(:,kj,kiter))*A(:,:,kj,kiter);
        end;
        
        %% f objective function
        
        %% p
        % p(x|c mu Lambda)
        log_px_tmp = zeros(N,1);
        log_px_tmp2 = zeros(K(kK),1);
        for kj = 1:K(kK)
            logdetB(kj) = 2*sum(log(diag(chol(B(:,:,kj,kiter)))));
            log_px_tmp = log_px_tmp - ...
                0.5*phi(:,kj,kiter).*...
                (diag((X'-ones(N,1)*mu(:,kj,kiter)')*(a(kj,kiter)*inv(B(:,:,kj,kiter)))*(X-mu(:,kj,kiter)*ones(1,N))) + ...
                trace((a(kj,kiter)*inv(B(:,:,kj,kiter)))*A(:,:,kj,kiter)));
            log_px_tmp2 (kj) = ...
                0.5*sum(n(kj,kiter)*(d*(d-1)/4*log(pi)+d*log(2)+sum(psi(a(kj,kiter)/2+(1-(1:d))/2))-logdetB(kj)));
        end;
        
        log_px(kiter) = -d/2*N*log(2*pi)+sum(log_px_tmp) + sum(log_px_tmp2);
        
        % p(c|pi)
        log_pc_tmp = zeros(N,1);
        for kj = 1:K(kK)
            % logdetA(kj) = 2*sum(log(diag(chol(A(:,:,kj,kiter)))));
            log_pc_tmp = log_pc_tmp + ...
                phi(:,kj,kiter)*(psi(alpha(kj,kiter))-psi(sum(alpha(:,kiter))));
            
        end;
        
        log_pc(kiter) = sum(log_pc_tmp);
        
        % p(pi)
        log_ppi(kiter) = gammaln(sum(alpha(:,kiter)))-sum(gammaln(alpha(:,kiter))) + ...
            sum((alpha(:,kiter)-1).*(psi(alpha(:,kiter))-psi(sum(alpha(:,kiter)))));
        
        % p(Lambda)
        for kj = 1:K(kK)
            psi_d(kj) = d*(d-1)/4*log(pi)+sum(psi(a(kj,kiter)/2+(1-(1:d))/2));
            trBLambda(kj) = sum(trace(B0*a(kj,kiter)*inv(B(:,:,kj,kiter))));
        end;
        
        logGamma_d0 = d*(d-1)/4*log(pi)+sum(gammaln(a0/2+(1-(1:d))/2));
        
        log_plambda(kiter) = (a0-d-1)/2*sum(psi_d+d*log(2)-logdetB)-...
            0.5*sum(trBLambda)-...
            a0/2*(d*log(2)+logdetB0)*K(kK)-logGamma_d0*K(kK);
        
        % p(mu)
        for kj = 1:K(kK)
            log_pmu_tmp(kj) = sum(diag(A(:,:,kj,kiter)+mu(:,kj,kiter)*mu(:,kj,kiter)'));
        end;
        log_pmu(kiter) = -K(kK)*(log(2*pi)+0.5*log(c))-1/2/c*sum(log_pmu_tmp);
        
        % p
        log_p(kiter) = log_px(kiter) + log_pc(kiter) + log_ppi(kiter) + log_plambda(kiter) + log_pmu(kiter);
        
        %% entropy
        log_q = zeros(Niter,1);
        % wishart
        for kj = 1:K(kK)
            logGamma_d(kj) = d*(d-1)/4*log(pi)+sum(gammaln(a(kj,kiter)/2+(1-(1:d))/2));
            log_q(kiter) = log_q(kiter) - (d+1)/2*logdetB(kj) + d*(d+1)/2*log(2) + logGamma_d(kj) - (a(kj,kiter)-d-1)/2*psi_d(kj) + a(kj,kiter)*d/2;
        end;
        % normal
        for kj = 1:K(kK)
            logdetA(kj) = 2*sum(log(diag(chol(A(:,:,kj,kiter)))));
            log_q(kiter) = log_q(kiter) + d/2*(1+log(2*pi)) + 0.5*logdetA(kj);
        end;
        % multinomial
        log_q(kiter) = log_q(kiter) + nansum(-nansum(phi(:,:,kiter).*log(phi(:,:,kiter)),2));
        % dirichlet
        log_q(kiter) = log_q(kiter) -  gammaln(sum(alpha(:,kiter))) + sum(gammaln(alpha(:,kiter))) + ...
            (sum(alpha(:,kiter))-K(kK))*psi(sum(alpha(:,kiter))) - sum((alpha(:,kiter)-1).*psi(alpha(:,kiter)));
        log_p_q(kiter) = log_p(kiter)+log_q(kiter);
    end;
    
    figure(h1);
    subplot(length(K),1,kK)
    plot(log_p_q(2:end))
    xlabel('Iteration')
    ylabel('Objective Function')
    title(['VI-GMM K=',num2str(K(kK))])
    
    [~,phi_max(:,kK)] = max(phi(:,:,Niter),[],2);
    
    if d<=3
        figure(h2);
        subplot(length(K),1,kK);
        for ksc = 1:K(kK)
            hold on;
            if d==2
                scatter(X(1,phi_max(:,kK)==ksc),X(2,phi_max(:,kK)==ksc))
            elseif d==3
                scatter3(X(1,phi_max(:,kK)==ksc),X(2,phi_max(:,kK)==ksc),X(2,phi_max(:,kK)==ksc))
            end;
        end;
    end;
    
    title(['VI-GMM K=',num2str(K(kK)),';effective K=',num2str(length(unique(phi_max(:,kK))))])
end;

