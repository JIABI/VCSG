%% for slt
function net = net_init_vgg(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

rng('default');
rng(2) ;

f=1/100;
net.layers = {} ;
% Block 1    

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(11,11,3,64, 'single'), zeros(1, 64, 'single') }}, ...
                           'stride',[4,4], ...
                           'pad',[0,0,0,0]) ;
                   
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.8) ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[2,2],'pad', [0,1,0,1]) ;
% Block 2

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,64,96, 'single'), zeros(1,96, 'single')}}, ...
                           'stride',[2,2], ...
                           'pad',[2,2,2,2]) ;
                       
net.layers{end+1} = struct('type', 'relu') ; 
%net.layers{end+1} = struct('type', 'bnorm') ; 
 net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.8) ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', [2,2],'pad', [0,1,0,1]) ;

% Block 3 

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,96,128 ,'single'), zeros(1,128, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad',[1,1,1,1]) ;
                      
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;


% Block 5
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,128,128, 'single'), zeros(1,128, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad',[1,1,1,1]) ;                    
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,128,256, 'single'), zeros(1,256, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
%net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[2,2],'pad', [1,1,1,1]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(2,2,256,512, 'single'), zeros(1,512, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad',0) ;                      
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
 net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,512, 'single'), zeros(1,512, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', 0) ;                     
net.layers{end+1} = struct('type', 'relu') ; 
%net.layers{end+1} = struct('type', 'bnorm') ; 
net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,100, 'single'), zeros(1,100, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;                        
net.layers{end+1} = struct('type', 'softmaxloss') ;