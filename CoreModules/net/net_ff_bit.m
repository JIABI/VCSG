function [ net,res,opts ] = net_ff_bit( net,res,opts )
%NET_FF Summary of this function goes here
%   Detailed explanation goes here

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    
    res(1).x=cast(res(1).x,opts.datatype);
    
    if opts.use_gpu
        res(1).x=gpuArray(res(1).x);
    end
    for layer=1:numel(net.layers)
%       if isfield(net.layers{1,layer},'weights')
%           net.layers{1,layer}.weights{1,10}=sign(net.layers{1,layer}.weights{1,1});  
% %            net.layers{1,layer}.weights{1,10}= net.layers{1,layer}.weights{1,10}/1;
%       end
        %disp(num2str([layer, size(res(layer).x)]))
        opts.current_layer=layer;
    
        switch net.layers{layer}.type

            case {'conv' , 'conv2d'}
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
                else
                   net.layers{layer}.stride=1;
                end
            if any(layer==1) 
                
            [res(layer+1).x,~,~,opts] = conv_layer_2d( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,[],opts );   
            else
                
              mask_pos=(net.layers{layer}.weights{1}>=max(0,min(1,(net.layers{layer}.weights{1})/2)));          
              mask_neg=(net.layers{layer}.weights{1}<max(0,min(1,(net.layers{layer}.weights{1})/2)));
              mask_neg=mask_neg*-1;
              net.layers{layer}.weights{10}=single(mask_pos+mask_neg);
%               elseif net.layers{layer}.weights{1}==1-max(0,min(1,(1+net.layers{layer}.weights{1})/2))
%                   net.layers{layer}.weights{10}=-1;                  
%               end  
%             net.layers{1,layer}.weights{1,10}=sign(net.layers{1,layer}.weights{1,1}); 
%             net.layers{1,layer}.weights{1,10}=logical(net.layers{1,layer}.weights{1,10});
%             [res(layer+1).x,~,~,opts] = conv_layer_2d( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,[],opts );   
            [res(layer+1).x,~,~,opts] = conv_layer_2d( res(layer).x,net.layers{1,layer}.weights{1,1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,[],opts );
%             [net,res(layer+1).x,~,~,opts] = bnorm( net,res(layer).x,layer,[],opts );
            end
%             net.layers{layer}.weights{1}=max(-1,min(1,net.layers{layer}.weights{1}));
            case 'conv1d'
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,2)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                
                if ~isfield(net.layers{layer},'stride')
                   net.layers{layer}.stride=1;
                end
                [res(layer+1).x,~,~,opts] = conv_layer_1d( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},net.layers{layer}.stride,net.layers{layer}.pad,[],opts );
   
            case {'mlp','linear'} 
                [res(layer+1).x,~,~,opts] = linear_layer( res(layer).x,net.layers{layer}.weights{1},net.layers{layer}.weights{2},[], opts );
                
            case 'dropout'
                if opts.training
                    dropout_opts.rate=net.layers{layer}.rate;
                    dropout_opts.mask=[];
                    [res(layer+1).x,dropout_opts.mask]= dropout(res(layer).x,[],dropout_opts );
                    net.layers{layer}.opts=dropout_opts;
                else
                    res(layer+1).x=res(layer).x;
                end
                
                
            case 'bnorm'
                [net,res(layer+1).x,~,~,opts] = bnorm( net,res(layer).x,layer,[],opts );
            case 'rmsnorm'
                [net,res(layer+1).x,opts] = rmsnorm( net,res(layer).x,layer,[],opts );                
            case {'normalize', 'lrn'}
                [res(layer+1).x,opts] = lrn(res(layer).x, net.layers{layer}.param(1),net.layers{layer}.param(2),net.layers{layer}.param(3),net.layers{layer}.param(4),[],opts) ;
            
            case 'relu'
                res(layer+1).x = relu(res(layer).x,[] );
            case 'modu'
                res(layer+1).x = modu(res(layer).x,[] );
            case 'leaky_relu'
                res(layer+1).x = leaky_relu(res(layer).x,[] );
            case 'sigmoid'
                res(layer+1).x = sigmoid_ln(res(layer).x,[] );
            case 'tanh'
                res(layer+1).x = tanh_ln(res(layer).x,[] );
            
            case 'pad'
                res(layer+1).x = pad_data(res(layer).x,net.layers{layer}.pad,[]);

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
                
                if opts.training==1
                    [res(layer+1).x,res(layer+1).from,opts] = maxpool(res(layer).x,net.layers{layer}.pool,net.layers{layer}.stride,net.layers{layer}.pad,[],[],opts);
                else
                    [res(layer+1).x,~,opts] = maxpool(res(layer).x,net.layers{layer}.pool,net.layers{layer}.stride,net.layers{layer}.pad,[],[],opts);
                end
                
            case 'pool1d' 
                
                if isfield(net.layers{layer},'pad')
                    if(length(net.layers{layer}.pad)==1)
                        net.layers{layer}.pad=ones(1,2)*net.layers{layer}.pad;
                    end
                else
                   net.layers{layer}.pad=[];
                end
                
                if opts.training==1
                    [res(layer+1).x,res(layer+1).from,opts] = maxpool_1d(res(layer).x,net.layers{layer}.pool,net.layers{layer}.stride,net.layers{layer}.pad,[],[],opts);
                else
                    [res(layer+1).x,~,opts] = maxpool_1d(res(layer).x,net.layers{layer}.pool,net.layers{layer}.stride,net.layers{layer}.pad,[],[],opts);
                end
            
            case 'softmax'        
                res(layer+1).x = softmax(res(layer).x,[]) ;

            case 'softmaxloss'
                res(layer+1).x = softmaxlogloss(res(layer).x, res(1).class) ;
                
%                     if any(layer~=1) 
%                             if isfield(net.layers{1,layer},'weights')
%                     if res(layer+1).x==max(0,min(1,(1+net.layers{layer}.weights{1})/2))
%                         net.layers{layer}.weights{10}=1;                 
%                       elseif res(layer+1).x==1-max(0,min(1,(1+net.layers{layer}.weights{1})/2))
%                          net.layers{layer}.weights{10}=-1;
%                     end
%                             end
%                     end
                %max(res(layer+1).x(:))
        
            otherwise 
                error('net_ff error')
                    
                    
        end


    end

%     for i=1:numel(net.layers)-1
%         figure;plot(reshape(gather(res(i).x),numel(res(i).x)./opts.parameters.batch_size,[]));title(['x to ',net.layers{i}.type,' (current layer)'])
%     end
end