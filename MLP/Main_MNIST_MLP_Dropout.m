clear all;close all;
addpath(genpath('../CoreModules'));
n_epoch=20; %training epochs
dataset_name='mnist'; %dataset name
network_name='mlp'; %network name
use_gpu=1 ;%use gpu or not 
opts.use_nntoolbox=1;
%function handle to prepare your data
lamda=[0 0.01 0.04 0.05 0.06 2 0.08 0.09 0.1 0.2 0.3 0.4];
n=length(lamda);
for myi=6:n
%function handle to prepare your data
PrepareDataFunc=@PrepareData_MNIST_MLP;
%function handle to initialize the network
NetInit=@net_init_mlp_mnist;
%automatically select learning rates
use_selective_sgd=0;
%select a new learning rate every n epochs
ssgd_search_freq=10;
learning_method=@sgd;%training method: @sgd,@adagrad,@rmsprop,@adam,@sgd2
sgd_lr=5e-2;
opts.sgd_lr_1=0.001;
opts.sgd_lr_2=0.001;
opts.sgd_lr_3= 0.001;
opts.alpha=0.80;
opts.myi=myi;
opts.parameters.u=0;
opts.parameters.lamda=lamda(myi);
Main_Template(); %call training template
%(unnecessary if selective-sgd is used)
opts.parameters.clip=1e-2;
opts.parameters.weightDecay=0;
opts.parameters.batch_size=500;
% 
% 
% Main_Template(); %call training template
if ~opts.LoadResults
    TrainingScript();
end

opts.TotalMemory(myi)=266200;
  if myi==1
       Pruning_sgd(myi)=opts.TotalMemory(myi);
   else
   Pruning_sgd(myi)=opts.pruning_sgd;
   end
   testError_km_p_svd_min(opts.myi)=min(opts.results.TestEpochError); 
   testError_km_p_svd_top5_min(opts.myi)=min(opts.results.TestEpochError_Top5); 
   testError_km_p_svd(opts.myi)=opts.results.TestEpochError(end); 
   testError_km_p_svd_top5(opts.myi)=opts.results.TestEpochError_Top5(end); 
   PRate_l1norm(myi)=(opts.TotalMemory(myi)-Pruning_sgd(myi))/opts.TotalMemory(myi);
   disp(PRate_l1norm);
%   trainLoss(opts.myi)=opts.results.TrainEpochLoss(end);
%   testLoss(opts.myi)=opts.results.TestEpochLoss(end);
end
memory=Pruning_sgd;