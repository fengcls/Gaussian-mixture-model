close all;
clear;clc

%% synthetic 2D data
rng(1,'twister')
mu1 = [6,8];
sigma1 = [1,1.5;1.5,3];
X1 = mvnrnd(mu1,sigma1,100);
mu2 = [0,4];
sigma2 = [3,-1.5;-1.5,1];
X2 = mvnrnd(mu2,sigma2,100);
mu3 = [4,2];
sigma3 = [2,0.5;0.5,1];
X3 = mvnrnd(mu3,sigma3,100);
X = [X1;X2;X3];

figure;scatter(X1(:,1),X1(:,2));hold on;
scatter(X2(:,1),X2(:,2));
scatter(X3(:,1),X3(:,2));
axis square;

%% run
cluster_em = gmm_em(X,[2,3,4]);
cluster_vi = gmm_vi(X,[2,3,4]);
cluster_gs = gmm_gs(X);
