clear all; 
close all;
% lambda_x=[0.2 0.4 0.5 0.6 0.8 0.95];
lambda_i=1;
% for lambda_i=1:length(lambda_x)    
addpath(genpath('/home/bonnie/LightNet-master/'));

n_epoch=325;
dataset_name='cifar';
network_name='cnn';
use_gpu=1;%use gpu or not ;
opts.use_nntoolbox=1; 

% % if use_gpu
% %     %Requires Neural Network Toolbox to use it.
% %     opts.use_nntoolbox=license('test','neural_network_toolbox');
% % end%function handle to prepare your data
% opts.parameters.l1_lr_3=lambda_x(lambda_i);    
%  lamda=[0 0 5 10 15 20 25 26 27 28 29 30 35];
 %parametersl1_lr_3=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
 lamda=[0 8 150 200 250 300 350 400 450 500 600 700 800 900];
% if use_gpu
%     %Requires Neural Network Toolbox to use it.
%     opts.use_nntoolbox=license('test','neural_network_toolbox')
% end

%  lamda=lamda/10;
%   lamda=[0 0.3 0.1 0.5 0.6 0.7 0.8 0.9 1 1.5 3];
% fcc_bits=[32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 ...
%    30 30 30 30 30 30 30 30 30 30 30 30 30 30 30 30  ...
%    28 28 28 28 28 28 28 28 28 28 28 28 28 28 28 28  ...
%    26 26 26 26 26 26 26 26 26 26 26 26 26 26 26 26  ...
%    24 24 24 24 24 24 24 24 24 24 24 24 24 24 24 24  ...
%    22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 22 ...
%    20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 ...
%    18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 18 ...
%    16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 ...
%     14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 ...
%     12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 12 ...
%     10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 10 ...
%     8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8   ...
%     7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 ...
%     6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 ...
%     5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 ...
%     4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 ...
%     3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3  ...
%     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 ];
% cnn_bits=[32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2 ...
%     32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2];
% fcc_bits=randi(32,1,n_epoch);
% fcc_bits=sort(fcc_bits,'descend');
% fcc_bits=[32 30 28 26 24 22 20 18 16 14 12 10 8 7 6 5 4 3 2];
%fcc_bits=[31 27 25 23 21 19 17 15 13 11 9 7 5];
%cnn_bits=[31 27 25 23 21 19 17 15 13 11 9 7 5];
%n=length(fcc_bits
nleng=length(lamda);
pruning_number=zeros(1,nleng);
total_number=zeros(1,nleng);
for myi=1:nleng
% opts.parameters.l1_lr_3=parametersl1_lr_3(myi); 
opts.grad1=0;
%PrepareDataFunc=@PrepareData_CIFAR_MLP;
%PrepareDataFunc=@PrepareData_imganet;
% PrepareDataFunc=@PrepareData_MNIST_MLP;
 PrepareDataFunc=@PrepareData_CIFAR_CNN;
%function handle to initialize the network
%load('/home/bonnie/LightNet-master/Alex-net.mat');
%NetInit=@net_init_cifar_mcnn;
 NetInit=@net_init_vgg;
% NetInit=@net_init_alexnet;
%  NetInit=@net_init_mlp_mnist_dropout;
%automatically select learning rates
use_selective_sgd=0;
%select a new learning rate every n epochs
ssgd_search_freq=10;
learning_method=@sgd2_SVRG;%training method: @sgd,@adagrad,@rmsprop,@adam,@sgd2 
% opts.l1=0.1;
% opts.l2=0.1;
sgd_lr=0.02;
%%imagenet
% opts.sgd_lr_1=500;
% opts.sgd_lr_2=690;
%cifar
% opts.sgd_lr_1=500;
% opts.sgd_lr_2=900;
opts.sgd_lr_1=0.15;
opts.sgd_lr_4=0.045; %0.45
opts.sgd_lr_2=0.5;
%imagnet 120;
opts.sgd_lr_3=0.009;
opts.parameters.l1_lr_6=0.8;
opts.parameters.l1_lr_4=0;
opts.parameters.batch_size=800;
opts.Dzdw=0;

%0.293;
%%mnist
% opts.sgd_lr_1=78;
% opts.sgd_lr_2=50;
opts.lr4_0=1;
opts.alpha=0.80;
opts.parameters.mom=0.9;
opts.myi=myi;
% opts.use_bnorm=1;
opts.parameters.u=0;
 opts.parameters.lamda=lamda(myi);
% opts.parameters.batch_size=0;
% opts.ppos_bias=0;
% opts.nneg_bias=015
% opts.ppos_bias=single(opts.ppos_bias);
% opts.nneg_bias=single(opts.nneg_bias);
   
% opts.parameters.weightDecay=0.9;
% opts.fcc_bits=fcc_bits(myi);
% opts.cnn_bits=cnn_bits(myi);
Main_Template(); %call training template
% opts.train=reshape(opts.train,[28 28 1 60000]);
% opts.test=reshape(opts.test,[28 28 1 10000]);
% opts.train_labels=single(opts.train_labels);
% opts.test_labels=single(opts.test_labels);
if ~opts.LoadResults
    TrainingScript();
end
for layer=1:numel(net.layers)
if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
total_number(opts.myi)=total_number(opts.myi)+size(net.layers{layer}.weights{1},1)*size(net.layers{layer}.weights{1},2);
pruning_number(opts.myi)=pruning_number(opts.myi)+nnz(net.layers{layer}.weights{1});
end
end

%  opts.TotalMemory(myi)=728416; %cifar-100;
% 
%    Pruning_sgd(lambda_i,1)=opts.TotalMemory(myi);
%    Pruning_sgd(lambda_i,myi)=opts.pruning_sgd;
%    testError_km_p_svd_min(opts.myi)=min(opts.results.TestEpochError); 
%    testError_km_p_svd_top5_min(opts.myi)=min(opts.results.TestEpochError_Top5); 
%    testError_km_p_svd(lambda_i,opts.myi)=opts.results.TestEpochError(end); 
%    testError_km_p_svd_top5(lambda_i,opts.myi)=opts.results.TestEpochError_Top5(end); 
%    PRate_l1norm(lamopts.results.TestEpochError(end);
opts.fx=0;
end
% bda_i,myi)=(opts.TotalMemory(myi)-Pruning_sgd(myi))/opts.TotalMemory(myi);
% %    disp(PRate_l1norm);
% % testError(opts.myi)=
% plot(testError,pruning_number);
% end
% use_gpu=0;%use gpu or not 
% opts.use_nntoolbox=0; %Requires Neural Network Toolbox to use it.
%  opts.parameters.current_ep=1;
%  start_ep=opts.parameter
% 
% for ep=start_ep:opts.n_epoch
%     opts.ep=ep;
% 
%     [net,opts]=train_net(net,opts);  
%     if isfield(opts,'valid')&&(numel(opts.valid)>0)
%         opts.validating=1;
%         [opts]=test_net(net,opts);
%     end
%     
%     for layer=1:numel(net.layers)
%       if isfield(net.layers{1,layer},'weights')
% %           net.layers{1, layer}.weights{1, 1}=int8(net.layers{1, layer}.weights{1, 1});
% %           net.layers{1, layer}.weights{1, 2}=int8(net.layers{1, layer}.weights{1, 2});
% %           bits_size=fi(net.layers{1, layer}.weights{1, 1},1,7,6);
% %           net.layers{1, layer}.weights{1, 1}=bits_size.single;
%     pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1});
%       end
%     end
%     opts.pruning_sgd=sum(pruning_sgd,2);
%     
%     opts.validating=0;
%     [opts]=test_net(net,opts);
%     
%     if opts.plot
%         %figure(figure1);
%         if strcmp(net.layers{end}.type,'softmaxloss')
%             subplot(1,2,1); 
%             plot(opts.results.TrainEpochError,'b','DisplayName','Train (top1)');hold on;
%             plot(opts.results.TrainEpochError_Top5,'b--','DisplayName','Train (top5)');hold on;
%             if isfield(opts,'valid')&&(numel(opts.valid)>0)
%                 plot(opts.results.ValidEpochError,'g','DisplayName','Valid (top1)');hold on
%                 plot(opts.results.ValidEpochError_Top5,'g--','DisplayName','Valid (top5)');hold on;
%             end
%             plot(opts.results.TestEpochError,'r','DisplayName','Test (top1)');hold on;
%             plot(opts.results.TestEpochError_Top5,'r--','DisplayName','Test (top5)');hold off;
%             
%             title('Error Rate per Epoch');legend('show');
%             subplot(1,2,2); 
%             plot(opts.results.TrainEpochLoss,'b','DisplayName','Train');hold on;
%             if isfield(opts,'valid')&&(numel(opts.valid)>0)
%                 plot(opts.results.ValidEpochLoss,'g','DisplayName','Valid');hold on;            
%             end            
%             plot(opts.results.TestEpochLoss,'r','DisplayName','Test');hold off;
%             title('Loss per Epoch');legend('show')
%             drawnow;
%         end
%         
%     end
%     
%     parameters=opts.parameters;
%     results=opts.results;
%     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
%     
%     opts.parameters.current_ep=opts.parameters.current_ep+1;
%     
% end
% 
% opts.train=[];
% opts.test=[];
% 
% if strcmp(net.layers{end}.type,'softmaxloss')
%     if isfield(opts,'valid')&&(numel(opts.valid)>0)
%         [min_err_valid,best_id]=min(opts.results.ValidEpochError);
%          min_err=opts.results.TestEpochError(best_id);  
%          disp(['Model validation error rate: ',num2str(min_err_valid)]);
%     else
%         [min_err,best_id]=min(opts.results.TestEpochError);
%     end
%     disp(['Model test error rate: ',num2str(min_err)]);
%     best_net_source=[fullfile(opts.output_dir,[opts.output_name,num2str(best_id),'.mat'])];
%     best_net_destination=[fullfile(opts.output_dir,['best_',opts.output_name,num2str(best_id),'.mat'])];
%     copyfile(best_net_source,best_net_destination);
% end
% 
% saveas(gcf,[fullfile(opts.output_dir,[opts.output_name,num2str(opts.n_epoch),'.pdf'])])
% 
% memory(1)=952600;
% testError_km_p_svd(1)=0.0799;
% testError_km_p_svd_top5(1)=0.056;