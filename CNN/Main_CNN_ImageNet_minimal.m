%function Main_CNN_ImageNet_minimal()
% Minimalistic demonstration of how to run an ImageNet CNN model
addpath(genpath('/home/bonnie/LightNet-master'));

n_epoch=400;
dataset_name='cifar';
network_name='cnn';
use_gpu=1;%use gpu or not ;
opts.use_nntoolbox=1; 

% setup toolbox
% addpath(genpath('../CoreModules'))
% download a pre-trained CNN from the web
if ~exist('imagenet-vgg-f.mat', 'file')
  fprintf('Downloading a model ... this may take a while\n') ;
%   urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-resnet-101-dag.mat', ...
%   'imagenet-resnet-101-dag.mat');
  urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
    'imagenet-vgg-f.mat') ;
end
%net=load('imagenet-resnet-101-dag.mat');
net=load('imagenet-vgg-f.mat') ;

% obtain and preprocess an image
% im = imread('test_im.JPG') ;
% im=imread('/home/bonnie/imagenet/val_data/n01440764_148.JPEG');
% im_ = single(im) ; % note: 255 range
% im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% im_ = im_ - net.meta.normalization.averageImage ;

use_selective_sgd=0;
%select a new learning rate every n epochs
ssgd_search_freq=10;
learning_method=@sgd2;%training method: @sgd,@adagrad,@rmsprop,@adam,@sgd2
% opts.l1=0.1;
% opts.l2=0.1;
sgd_lr=0.03;
opts.sgd_lr_1=900;
opts.sgd_lr_2=550;
% opts.sgd_lr_1=1;
% opts.sgd_lr_2=3;
opts.lr4_0=1;
opts.alpha=0.80;
opts.parameters.mom=0.9;
opts.myi=myi;
opts.parameters.u=0;
opts.parameters.lamda=lamda(myi);
opts.parameters.batch_size=700;
PrepareDataFunc=@PrepareData_CIFAR_CNN;
Main_Template();
% run the CNN
opts=[];
opts.use_gpu=1;%unless you have a good gpu
opts.use_nntoolbox=0; %Requires Neural Network Toolbox to use it.

opts.training=0;
opts.use_corr=1;
res(1).x=im_;

if opts.use_gpu
    net=SwitchProcessor(net,'gpu');
end
tic;
[ net,res,opts ] = net_ff( net,res,opts );
toc;
% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
   net.meta.classes.description{best}, best, bestScore)) ;
drawnow;

