%function net = net_init_cifar_cnn(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST


% rng('default'); 
% rng(0) ; 
% f=1/100; 
% net.layers = {} ; % Block 1
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*(-6+12*rand(5,5,3,32, 'single')), zeros(1, 32, 'single') }}, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
%                            
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad',[1,1,1,1]) ; % Block 2
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*(-5+10*rand(5,5,32,96,'single')), zeros(1,96, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad',[1,1,1,1]) ;
% 
% % Block 3
% 
% net.layers{end+1} = struct('type', 'conv',...
%                             'weights', {{f*(-4+8*rand(5,5,96,96,'single')), zeros(1,96, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad',[1,1,1,1]) ;
% 
% % Block 4 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*(-5+10*rand(4,4,96,256, 'single')), zeros(1,256, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;
% % Block 5
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*(5+10*rand(1,1,256,100,'single')), zeros(1, 100, 'single')}}, ...
%                            'stride', 1, ... 
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'softmaxloss') ;

% % % % % % rng('default');
% % % % % % rng(0) ;
% % % % % % 
% % % % % % %f=1/100 ;
% % % % % % f=1/100;
% % % % % % 
% % % % % % net.layers = {} ;
% % % % % % % Block 1  96  
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(11,11,3,96, 'single'), zeros(1, 96, 'single') }}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 2) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% % % % % % % Block 2 26
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(5,5,96,64, 'single'), zeros(1,64, 'single')}}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 2) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(5,5,64,32, 'single'), zeros(1,32, 'single')}}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 2) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% % % % % % 
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(5,5,32,32, 'single'), zeros(1,32, 'single')}}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 2) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% % % % % % % Block 3 8
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(5,5,32,64, 'single'), zeros(1,64, 'single')}}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 2) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% % % % % % 
% % % % % % % Block 4  6
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(4,4,64,64, 'single'), zeros(1,64, 'single')}}, ...
% % % % % %                            'stride', 2, ...
% % % % % %                            'pad', 1) ;
% % % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % % %net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', [0,0,0,0]) ;
% % % % % % % Block 5
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % % %                            'weights', {{f*randn(1,1,64,10, 'single'), zeros(1, 10, 'single')}}, ...
% % % % % %                            'stride', 1, ...
% % % % % %                            'pad', 0) ;
% % % % % % 
% % % % % % net.layers{end+1} = struct('type', 'softmaxloss') ;


% rng('default');
% rng(0) ;
% 
% %f=1/100 ;
% f=1/100;
% 
% net.layers = {} ;
% % Block 1  96  
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(11,11,3,96, 'single'), zeros(1, 96, 'single') }}, ...
%                            'stride', 2, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;
% % Block 2 26
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,96,64, 'single'), zeros(1,64, 'single')}}, ...
%                            'stride', 2, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;
% 
% % Block 3 8
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,64,64, 'single'), zeros(1,64, 'single')}}, ...
%                            'stride', 2, ...
%                            'pad', 2) ;
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', 1) ;
% net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;
% 
% % Block 4  6
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(4,4,64,128, 'single'), zeros(1,128, 'single')}}, ...
%                            'stride', 2, ...
%                            'pad', 1) ;
% net.layers{end+1} = struct('type', 'relu') ;
% %net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', [0,0,0,0]) ;
% % Block 5
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,128,10, 'single'), zeros(1, 10, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% 
% net.layers{end+1} = struct('type', 'softmaxloss') ;
% % 

function net = net_init_cifar_mcnn(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST


rng('default');
rng(3) ;

f=1/100 ;
net.layers = {} ;
% Block 1    
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,3,32, 'single'), zeros(1, 32, 'single') }}, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', [1,1,1,1]) ;
% Block 2
%net.layers{end+1} = struct('type', 'dropout','rate',0.3) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,32,32, 'single'), zeros(1,32, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;

net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', [1,1,1,1]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.3) ;
% Block 3

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,32,64, 'single'), zeros(1,64, 'single')}}, ...
                           'stride', 1, ...
                           'pad',2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 2,'pad', [1,1,1,1]) ;
 net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;
% Block 4
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,64,64, 'single'), zeros(1,64, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
  net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;

% Block 5

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,64,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
% if opts.use_bnorm
% net.layers{end+1} = struct('type', 'bnorm') ; 
% end
net.layers{end+1} = struct('type', 'softmaxloss') ;


