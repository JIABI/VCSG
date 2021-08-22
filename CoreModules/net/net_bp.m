function [ net,res,opts ] = net_bp( net,res,opts)
%NET_FF Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    opts.dzdy=cast(opts.dzdy,opts.datatype);
    res(numel(net.layers)+1).dzdx = opts.dzdy ;
    if opts.use_gpu
         res(numel(net.layers)+1).dzdx=gpuArray( res(numel(net.layers)+1).dzdx);      
    end
    

        
    for layer=numel(net.layers):-1:1
          

        opts.current_layer=layer;
        switch net.layers{layer}.type

            case {'conv','conv2d'}
                if isfield(net.layers{layer},'stride')
                    if(length(net.layers{layer}.stride)==1)
                        net.layers{layer}.stride=ones(1,2)*net.layers{layer}.stride;
                    end
                else
                   net.layers{layer}.stride=1;
                end
                
                
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,4)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb,opts] = conv_layer_2d( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,res(layer+1).dzdx,opts);            
%                 if opts.mini_b==opts.n_train_batch
%                 [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb,opts] = conv_layer_2d( opts.res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,opts.res(layer+1).dzdx,opts);            
%                 opts.res(layer).dzdx=res(layer).dzdx;opts.res(layer).dzdw=res(layer).dzdw;opts.res(layer).dzdb=res(layer).dzdb;
%                 end
            case 'conv1d'
                if ~isfield(net.layers{layer},'stride')
                   net.layers{layer}.stride=1;
                end
                
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,2)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb,opts] = conv_layer_1d( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,res(layer+1).dzdx,opts );
             
                
            case {'mlp','linear'}                             
                [res(layer).dzdx, res(layer).dzdw,res(layer).dzdb] = linear_layer( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},res(layer+1).dzdx,opts );
            
            
            case 'dropout'
                [res(layer).dzdx,~]= dropout(res(layer).x,res(layer+1).dzdx,net.layers{layer}.opts);
%                 if opts.mini_b==opts.n_train_batch
%                 [res(layer).dzdx,~]= dropout(opts.res(layer).x,opts.res(layer+1).dzdx,net.layers{layer}.opts);
%                 opts.res(layer).dzdx=res(layer).dzdx;
%                 end
                                
            case 'bnorm'
                [~,res(layer).dzdx,res(layer).dzdw,res(layer).dzdb] = bnorm( net,res(layer).x,layer,res(layer+1).dzdx ,opts );                
            case 'rmsnorm'
                [net,res(layer).dzdx,opts] = rmsnorm( net,res(layer).x,layer,res(layer+1).dzdx ,opts );                
            case {'normalize', 'lrn'}
                res(layer).dzdx = lrn(res(layer).x, net.layers{layer}.param(1),net.layers{layer}.param(2),net.layers{layer}.param(3),net.layers{layer}.param(4),res(layer+1).dzdx,opts ) ;
            case 'relu'
                res(layer).dzdx = relu(res(layer).x, res(layer+1).dzdx) ;
%                 if opts.mini_b==opts.n_train_batch
%                 res(layer).dzdx = relu(opts.res(layer).x, opts.res(layer+1).dzdx) ;
%                 opts.res(layer).dzdx=res(layer).dzdx;
%                 end
            case 'modu'
                res(layer).dzdx = modu(res(layer).x, res(layer+1).dzdx) ;
            case 'leaky_relu'
                res(layer).dzdx = leaky_relu(res(layer).x, res(layer+1).dzdx) ;   
            case 'sigmoid'
                res(layer).dzdx = sigmoid_ln(res(layer).x,res(layer+1).dzdx );
            case 'tanh'
                res(layer).dzdx = tanh_ln(res(layer).x,res(layer+1).dzdx );
%                 if opts.mini_b==opts.n_train_batch
%                 res(layer).dzdx = tanh_ln(opts.res(layer).x,opts.res(layer+1).dzdx );
%                 opts.res(layer).dzdx=res(layer).dzdx;
%                 end
            case 'pad'
                [res(layer).x,res(layer).dzdx]=pad_data(res(layer+1).x,net.layers{layer}.pad,res(layer+1).dzdx);
%           if opts.mini_b==opts.n_train_batch
%                 [res(layer).x,res(layer).dzdx]=pad_data(opts.res(layer+1).x,net.layers{layer}.pad,opts.res(layer+1).dzdx);
%               opts.res(layer).x=res(layer).x;
%               opts.res(layer).dzdx=res(layer).dzdx;
%           end
            case 'pool'
                
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,4)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                                
                if isfield(net.layers{layer},'stride')
                    if(length(net.layers{layer}.stride)==1)
                        net.layers{layer}.stride=ones(1,2)*net.layers{layer}.stride;
                    end
                end
                
                res(layer).dzdx = maxpool(res(layer).x, net.layers{layer}.pool, net.layers{layer}.stride,net.layers{layer}.pad,res(layer+1).dzdx,res(layer+1).from,opts);
%                  if opts.mini_b==opts.n_train_batch
%                 res(layer).dzdx = maxpool(opts.res(layer).x, net.layers{layer}.pool, net.layers{layer}.stride,net.layers{layer}.pad,opts.res(layer+1).dzdx,opts.res(layer+1).from,opts);
%                  opts.res(layer).dzdx=res(layer).dzdx;
%                  end
            case 'pool1d'
                
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,2)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                
                res(layer).dzdx = maxpool_1d(res(layer).x, net.layers{layer}.pool, net.layers{layer}.stride,net.layers{layer}.pad,res(layer+1).dzdx,res(layer+1).from,opts);
%                  if opts.mini_b==opts.n_train_batch
%                  res(layer).dzdx = maxpool_1d(opts.res(layer).x, net.layers{layer}.pool, net.layers{layer}.stride,net.layers{layer}.pad,opts.res(layer+1).dzdx,opts.res(layer+1).from,opts);
%                  opts.res(layer).dzdx=res(layer).dzdx;
%                  end
                 
            case 'softmax'        
                res(layer).dzdx = softmax(res(layer).x,res(layer+1).dzdx) ;
%                  if opts.mini_b==opts.n_train_batch
%                 res(layer).dzdx = softmax(opts.res(layer).x,opts.res(layer+1).dzdx) ;
%                 opts.res(layer).dzdx =  res(layer).dzdx;
%                  end
            case 'softmaxloss'
                res(layer).dzdx = softmaxlogloss(res(layer).x, res(1).class, res(layer+1).dzdx) ;
%                 if opts.mini_b==opts.n_train_batch
%                 res(layer).dzdx = softmaxlogloss(opts.res(layer).x, opts.res(1).class, opts.res(layer+1).dzdx) ;
%                 opts.res(layer).dzdx=res(layer).dzdx;
%                 end
            otherwise 
                error('net_np error')

        end
        


    end

    %{
    %for visualization
    for i=1:numel(net.layers)-1
        figure;subplot(1,2,1);plot(reshape(gather(res(i).x),numel(res(i).x)./opts.parameters.batch_size,[]));title(['x to ',net.layers{i}.type,' (current layer)'])
        subplot(1,2,2);plot(reshape(gather(res(i+1).dzdx),numel(res(i+1).dzdx)./opts.parameters.batch_size,[]));title(['dzdy from ', net.layers{i+1}.type, ' (next layer)'])
    end
    %}
    
end

