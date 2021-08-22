%% for slt
function net = net_init_alexnet(opts)
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

rng('default');
rng(6) ;

f=1/10;
net.layers = {} ;
% Block 1    

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(11,11,3,32, 'single'), zeros(1, 32, 'single') }}, ...
                           'stride',[2,2], ...
                           'pad',[2,2,2,2]) ;
                   
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ;     
net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.8) ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[2,2],'pad', [0,1,0,1]) ;
% Block 2

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,32,96, 'single'), zeros(1,96, 'single')}}, ...
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
net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;


% Block 5
% % net.layers{end+1} = struct('type', 'conv', ...
% %                            'weights', {{f*randn(3,3,64,64, 'single'), zeros(1, 64, 'single')}}, ...
% %                            'stride', [1,1], ...
% %                            'pad',[1,1,1,1]) ;                    
% % net.layers{end+1} = struct('type', 'relu') ;
% % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,128,128, 'single'), zeros(1,128, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', [1,1,1,1]) ;
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
%net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
%net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;
net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[2,2],'pad', [1,1,1,1]) ;


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(2,2,128,1024, 'single'), zeros(1,1024, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad',0) ;                      
net.layers{end+1} = struct('type', 'relu') ;
%net.layers{end+1} = struct('type', 'bnorm') ; 
net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,1024,1024, 'single'), zeros(1,1024, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', 0) ;                     
net.layers{end+1} = struct('type', 'relu') ; 
%net.layers{end+1} = struct('type', 'bnorm') ; 
net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;


net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,1024,10, 'single'), zeros(1,10, 'single')}}, ...
                           'stride', [1,1], ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

%% for cifar 
% % % % % function net = net_init_alexnet(opts)
% % % % % % CNN_MNIST_LENET Initialize a CNN similar for MNIST
% % % % % 
% % % % % rng('default');
% % % % % rng(6) ;
% % % % % 
% % % % % f=1/100;
% % % % % net.layers = {} ;
% % % % % % Block 1    
% % % % % 
% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(11,11,3,32, 'single'), zeros(1, 32, 'single') }}, ...
% % % % %                            'stride',1, ...
% % % % %                            'pad',2) ;
% % % % %                    
% % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % %net.layers{end+1} = struct('type', 'bnorm') ;     
% % % % % net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% % % % % net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;
% % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[2,2],'pad', [1,1,1,1]) ;
% % % % % % Block 2
% % % % % 
% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(5,5,32,32, 'single'), zeros(1,32, 'single')}}, ...
% % % % %                            'stride',1, ...
% % % % %                            'pad',2) ;
% % % % %                        
% % % % % net.layers{end+1} = struct('type', 'relu') ; 
% % % % % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % % % % net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% % % % % net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;
% % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', [2,2],'pad', [1,1,1,1]) ;

% % Block 3 
% 
% % net.layers{end+1} = struct('type', 'conv', ...
% %                            'weights', {{f*randn(3,3,32,64 ,'single'), zeros(1,64, 'single')}}, ...
% %                            'stride', 1, ...
% %                            'pad',1) ;                     
% % net.layers{end+1} = struct('type', 'leaky_relu') ;
% % net.layers{end+1} = struct('type', 'dropout','rate',0.5) ; 
%net.layers{end+1} = struct('type', 'bnorm') ; 
%


% % % Block 5
% % net.layers{end+1} = struct('type', 'conv', ...
% %                            'weights', {{f*randn(3,3,64,64, 'single'), zeros(1, 64, 'single')}}, ...
% %                            'stride', [1,1], ...
% %                            'pad',[1,1,1,1]) ;                    
% % net.layers{end+1} = struct('type', 'leaky_relu') ;
% % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % net.layers{end+1} = struct('type', 'dropout','rate',0.5) ;

% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(3,3,32,32, 'single'), zeros(1,32, 'single')}}, ...
% % % % %                            'stride', 1, ...
% % % % %                            'pad', 0) ;
% % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % % % % %net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% % % % % net.layers{end+1} = struct('type', 'dropout','rate',0.2) ;
% % % % % net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride',[1,1],'pad', [1,1,1,1]) ;
% % % % % 
% % % % % 
% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(5,5,32,64, 'single'), zeros(1,64, 'single')}}, ...
% % % % %                            'stride', 2, ...
% % % % %                            'pad',0) ;                  
% % % % % net.layers{end+1} = struct('type', 'relu') ;
% % % % % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % % % % net.layers{end+1} = struct('type', 'dropout','rate',0.3) ;   
% % % % % 
% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(1,1,64,64, 'single'), zeros(1,64, 'single')}}, ...
% % % % %                            'stride', [1,1], ...  
% % % % %                             'pad',0) ;   
% % % % % net.layers{end+1} = struct('type', 'relu') ; 
% % % % % %net.layers{end+1} = struct('type', 'bnorm') ; 
% % % % % net.layers{end+1} = struct('type', 'dropout','rate',0.3) ;  
% % % % % 
% % % % % net.layers{end+1} = struct('type', 'conv', ...
% % % % %                            'weights', {{f*randn(1,1,64,10, 'single'), zeros(1,10, 'single')}}, ...
% % % % %                            'stride', 1, ...
% % % % %                            'pad', 0) ;
% % % % % %net.layers{end+1} = struct('type', 'sigmoid') ;                       
% % % % % net.layers{end+1} = struct('type', 'softmaxloss') ;


% function net = net_init_alexnet(opts)
% % CNN_MNIST_LENET Initialize a CNN similar for MNIST
% leaky_relu
% rng('default');
% rng(0) ;
% 
% f=1/100;
% net.layers = {} ;
% % Block 1    
%  
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,3,96, 'single'), zeros(1, 96, 'single') }}, ...
%                            'stride',2, ...
%                            'pad',0) ;
%                    
% net.layers{end+1} = struct('type', 'relu') ;
%  %net.layers{end+1} = struct('type', 'bnorm') ;     
%  net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 4,'pad', 0) ;
% % Block 2
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(5,5,96,256, 'single'), zeros(1,256, 'single')}}, ...
%                            'stride',1, ...
%                            'pad',0) ;
%                        
% net.layers{end+1} = struct('type', 'relu') ; 
%  %net.layers{end+1} = struct('type', 'bnorm') ;
% net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 1,'pad', 0) ;
% 
% % Block 3 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(3,3,256,384 ,'single'), zeros(1,384, 'single')}}, ...
%                            'stride', 2, ...
%                            'pad', 1) ;
%                       
% net.layers{end+1} = struct('type', 'relu') ;
% %  net.layers{end+1} = struct('type', 'bnorm') ; 
% %  net.layers{end+1} = struct('type', 'lrn','param',[5 1 2.000000000000000e-05 0.75]) ;
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 1,'pad', 2) ;
% 
% % Block 4
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(3,3,384,384, 'single'), zeros(1,384, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 1) ;
% % net.layers{end+1} = struct('type', 'bnorm') ;                       
% net.layers{end+1} = struct('type', 'relu') ;
% 
% 
% % Block 5
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(3,3,384,256, 'single'), zeros(1, 256, 'single')}}, ...
%                            'stride',1, ...
%                            'pad', 0) ;
% % net.layers{end+1} = struct('type', 'bnorm') ;                       
% net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'pool','pool', 3, 'stride', 1,'pad', 0) ;
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,256,4096, 'single'), zeros(1,4096, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 1) ;                      
% net.layers{end+1} = struct('type', 'relu') ;
% % net.layers{end+1} = struct('type', 'pool','pool',3, 'stride', 10,'pad', [1,1,1,1]) ;
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,4096,4096, 'single'), zeros(1,4096, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 1) ;                     
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'conv', ...
%                            'weights', {{f*randn(1,1,4096,1000, 'single'), zeros(1,1000, 'single')}}, ...
%                            'stride', 1, ...
%                            'pad', 0) ;
% % net.layers{end+1} = struct('type', 'bnorm') ;                      
% net.layers{end+1} = struct('type', 'relu') ;
% 
% net.layers{end+1} = struct('type', 'softmaxloss') ;