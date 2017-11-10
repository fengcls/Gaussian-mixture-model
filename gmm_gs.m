function phi_max = gmm_gs(X,Niter)
%% GS-GMM
disp('===== Gibbs Sampling Gaussian Mixture Model =====')
rng(1,'twister')
if size(X,1)>size(X,2)
    X = X';
end;
N   = size(X,2);
d   = size(X,1);
if nargin==1
    Niter = 100;
end;

% mu_j | Lambda_j ~ Normal(m, inv(c*Lambda))
m = mean(X,2);
c = 0.1;

% Lambda_j ~ Wishart(a,B)
a = d;
A = cov(X');
B = c*d*A;

alpha = 1;

cind = ones(N,Niter);
Knum = ones(Niter,1);
Kmax = N+1;
phi = zeros(N,Kmax,Niter);
Lambda = zeros(d,d,Kmax,Niter);
mu = zeros(d,Kmax,Niter);

% note that the definition of wishart
Lambda(:,:,1,1) = wishrnd(inv(B),a);
mu(:,1,1) = mvnrnd(m',inv(c*Lambda(:,:,1,1)));

cindcount = zeros(6,Niter);

for kiter = 2:Niter
    disp(kiter);
    cind(:,kiter) = cind(:,kiter-1);
    mu(:,:,kiter) = mu(:,:,kiter-1);
    Lambda(:,:,:,kiter) = Lambda(:,:,:,kiter-1);
    for ki = 1:N
        clusternumtmp = max(cind(:,kiter));
        for kj = 1:clusternumtmp
            if sum(cind(setdiff(1:N,ki),kiter)==kj)
                phi(ki,kj,kiter) = ...
                    mvnpdf(X(:,ki)',mu(:,kj,kiter)',inv(Lambda(:,:,kj,kiter)))*... % here is the problem I misadd a c in the generating function p(x|c)
                    sum(cind(setdiff(1:N,ki),kiter)==kj)/(alpha+N-1);
            end;
        end;
        % new cluster
        kj = clusternumtmp+1;
        phi(ki,kj,kiter) = alpha/(alpha+N-1)*...
            (c/pi/(1+c))^(d/2)/(det(B))^(-a/2)*...
            det(B+c/(1+c)*(X(:,ki)-m)*(X(:,ki)-m)')^(-(a+1)/2)*...
            exp(sum(gammaln((a+1)/2+(1-(1:d))/2)-gammaln(a/2+(1-(1:d))/2)));
        
        % normalize the probability
        phi(ki,:,kiter) = phi(ki,:,kiter)/sum(phi(ki,:,kiter),2);
        
        % sample c index
        cind(ki,kiter) = sum(rand(1)>cumsum(phi(ki,:,kiter)))+1;
        
        % if selecting the new cluster, generate the associated
        % parameters
        if (cind(ki,kiter)==kj)
            s = 1;
            mp = c/(c+s)*m+1/(s+c)*X(:,ki);
            cp = c+s;
            ap = a+s;
            Bp = B+s/(a*s+1)*(X(:,ki)-m)*(X(:,ki)-m)';
            % wishart distribution
            Lambda(:,:,kj,kiter) = wishrnd(inv(Bp),ap);
            mu(:,kj,kiter) = mvnrnd(mp,inv(cp*Lambda(:,:,kj,kiter)));
        end;
    end;
    
    % reindex the cluster
    Knum(kiter) = max(cind(:,kiter));
    excludeind = [];
    for kj = 1:Knum(kiter)
        s = sum(cind(:,kiter)==kj);
        if ~s
            excludeind = [excludeind kj];
        else
            mp = c/(c+s)*m+1/(s+c)*sum(X(:,cind(:,kiter)==kj),2);
            cp = c+s;
            ap = a+s;
            Xmean = mean(X(:,cind(:,kiter)==kj),2);
            Bp = B+s/(a*s+1)*(Xmean-m)*(Xmean-m)'+(X(:,cind(:,kiter)==kj)-Xmean*ones(1,s))*(X(:,cind(:,kiter)==kj)-Xmean*ones(1,s))';
            % sample Lambda and mu
            
            % 20161109
            % Bp = (Bp+Bp')/2;
            Lambda(:,:,kj,kiter) = wishrnd(inv(Bp),ap);
            mu(:,kj,kiter) = mvnrnd(mp,inv(cp*Lambda(:,:,kj,kiter)));
        end;
    end;
    if ~isempty(excludeind)
        keepind = setdiff(1:Knum(kiter),excludeind);
        mutmp = mu(:,keepind,kiter);
        Lambdatmp = Lambda(:,:,keepind,kiter);
        Knum(kiter) = Knum(kiter) - length(excludeind);
        mu(:,1:Knum(kiter),kiter) = mutmp;
        Lambda(:,:,1:Knum(kiter),kiter) = Lambdatmp;
        
        cindtmp = cind(:,kiter);
        for k = 1:length(keepind)
            cindtmp(cindtmp == keepind(k))=k;
        end;
        cind(:,kiter) = cindtmp;
    end;
    
    counttmp = histc(cind(:,kiter),unique(cind(:,kiter)));
    if length(counttmp)>6;counttmp = counttmp(1:6);end;
    cindcount(1:length(counttmp),kiter) = counttmp;
    
end;
cindcount = sort(cindcount);
figure;plot(cindcount')
legend('1st','2nd','3rd','4th','5th','6th','Location','northeastoutside')
xlabel('iteration');
ylabel('#observations per cluster')

figure;plot(Knum)
xlabel('iteration');
ylabel('#clusters')

last_proportion = 0.5;
Knum_mode = mode(Knum(round(last_proportion*Niter):end));
phi_mean_knum_mode = mean(phi(:,:,Knum(round(last_proportion*Niter):end)==Knum_mode),3);
[~,phi_max] = max(phi_mean_knum_mode,[],2);

if d<=3
    figure;
    for ksc = 1:Knum_mode
        hold on;
        if d==2
            scatter(X(1,phi_max==ksc),X(2,phi_max==ksc))
        elseif d==3
            scatter3(X(1,phi_max==ksc),X(2,phi_max==ksc),X(2,phi_max==ksc))
        end;
    end;
end;

