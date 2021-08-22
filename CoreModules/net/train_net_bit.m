function [net,opts]=train_net_bit(net,opts)

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    
    opts.training=1;

%    if ~isfield(opts.parameters,'learning_method')
              
%    end
    
    if ~isfield(opts,'display_msg')
        opts.display_msg=1; 
    end
    opts.TrainMiniBatchError=[];
    opts.TrainMiniBatchError_Top5=[];
    opts.TrainMiniBatchLoss=[];
    
    
    tic
    
    opts.order=randperm(opts.n_train);    

    if opts.parameters.selective_sgd==1 
        [ net,opts ] = selective_sgd( net,opts );
    end
    
    batch_dim=length(size(opts.train));
    idx_nd=repmat({':'},[1,batch_dim]);
 
    for mini_b=1:opts.n_train_batch   
    nn1=1/(1+(mini_b/opts.n_epoch^(2)));
    opts.parameters.l1_lr_1=opts.sgd_lr_1*nn1;
    nn2=mini_b/opts.parameters.batch_size;
    opts.parameters.l1_lr_2=opts.sgd_lr_2*opts.alpha^(nn2);
    nn3=1/(1+0.7*(mini_b/opts.parameters.batch_size)^(3));
    opts.parameters.l1_lr_3=opts.sgd_lr_3*nn3; 
    opts.parameters.u=opts.parameters.u+opts.parameters.l1_lr_2*(opts.parameters.lamda/opts.parameters.batch_size);%
        
        idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);       
        idx_nd{end}=idx;
        opts.idx_nd=idx_nd;
        opts.idx=idx;
        res(1).x=opts.train(idx_nd{:});
        
        %classification
        if strcmp(net.layers{end}.type,'softmaxloss')
            res(1).class=opts.train_labels(idx);
        end
%         for layer=1:numel(net.layers)
% %         if isfield(net.layers{1,layer},'weights')
% %            net.layers{1,layer}.weights{1,10}=sign(net.layers{1,layer}.weights{1,1});  
% %            net.layers{1,layer}.weights{1,10}= net.layers{1,layer}.weights{1,10};
% %         end
%         end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        
        [ net,res,opts ] = net_ff_bit( net,res,opts );


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%backward%%%%%%%%%%%%%%%%
        opts.dzdy=1;
        [ net,res,opts ] = net_bp_bit( net,res,opts );
       

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%summarize the current batch
        loss=double(gather(mean(res(end).x(:))));
        
        if strcmp(net.layers{end}.type,'softmaxloss')
            err=error_multiclass(res(1).class,res);
            opts.TrainMiniBatchError=[opts.TrainMiniBatchError;err(1)/opts.parameters.batch_size];
            opts.TrainMiniBatchError_Top5=[opts.TrainMiniBatchError_Top5;err(2)/opts.parameters.batch_size];
            if opts.display_msg==1
                disp(['Minibatch loss: ', num2str(loss),...
                    ', top 1 err: ', num2str(opts.TrainMiniBatchError(end)),...
                    ',top 5 err:,',num2str(opts.TrainMiniBatchError_Top5(end))])
            end
        end
        
        opts.TrainMiniBatchLoss=[opts.TrainMiniBatchLoss;loss];                 
        if (~isfield(opts.parameters,'iterations'))
            opts.parameters.iterations=0; 
        end
        opts.parameters.iterations=opts.parameters.iterations+1;
        
        
%           counter=0;
%           for nlayer=1:numel(net.layers)
%         if isfield(net.layers{nlayer},'weights')
%               counter=counter+1; 
%         pos_bias_mask=net.layers{1,nlayer}.weights{1, 2}>0;
%         neg_bias_mask=net.layers{1,nlayer}.weights{1, 2}<0;
%         pos_bias=pos_bias_mask.*net.layers{1,nlayer}.weights{1, 2};      
%         min_pos_bias=pos_bias(find(pos_bias>0));
%         ppos_bias=min(min_pos_bias);
%         if isempty(ppos_bias) 
%             ppos_bias=0;
%         end
%         pppos_bias(1,counter)=ppos_bias;
%         neg_bias=neg_bias_mask.*net.layers{1,nlayer}.weights{1, 2};   
%         min_neg_bias=neg_bias(find(neg_bias<0));
%         nneg_bias=min(min_neg_bias); 
%         if isempty(nneg_bias) 
%             nneg_bias=0;
%         end
%         nnneg_bias(1,counter)=nneg_bias;
%         end
%           end
%         opts.ppos_bias=min(pppos_bias);
%         opts.ppos_bias=single(opts.ppos_bias);
%         opts.nneg_bias=min(nnneg_bias);
  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%stochastic gradients descent%%%%%%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = sgd_bit( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            for layer=1:numel(net.layers)
%            if isfield(net.layers{1,layer},'weights')
% %            if opts.parameters.iterations==1
% %                 net.layers{1, layer}.weights{1, 6}=0;
% %                 net.layers{1, layer}.weights{1, 7}=0;
% %                 net.layers{1, layer}.weights{1, 8}=0;
% %                 net.layers{1, layer}.weights{1, 9}=0;
% %            elseif opts.parameters.iterations==2
% %                 net.layers{1, layer}.weights{1, 6}=res(layer).dzdw;
% %                 net.layers{1, layer}.weights{1, 7}=net.layers{1, layer}.weights{1, 6};
% %                 net.layers{1, layer}.weights{1, 8}=res(layer).dzdb;
% %                 net.layers{1, layer}.weights{1, 9}=net.layers{1, layer}.weights{1, 8};
% %            else
%            net.layers{1, layer}.weights{1, 6}=(1/opts.n_train)*(net.layers{1, layer}.weights{1, 6}+res(layer).dzdw);
%            net.layers{1, layer}.weights{1, 7}=res(layer).dzdw;
%            net.layers{1, layer}.weights{1, 8}=(1/opts.n_train)*(net.layers{1, layer}.weights{1, 8}+res(layer).dzdb);
%            net.layers{1, layer}.weights{1, 9}=res(layer).dzdb;
%            end
%            end
             
     end   
        
    %%summarize the current epoch
     if ~isfield(opts,'results')||~isfield(opts.results,'TrainEpochLoss')
        opts.results.TrainEpochLoss=[];
        opts.results.TrainEpochError=[];
        opts.results.TrainEpochError_Top5=[];
     end
        
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.TrainMiniBatchLoss(:))];

    if strcmp(net.layers{end}.type,'softmaxloss')        
        opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.TrainMiniBatchError(:))];
        opts.results.TrainEpochError_Top5=[opts.results.TrainEpochError_Top5;mean(opts.TrainMiniBatchError_Top5(:))];
        disp(['Epoch ',num2str(opts.parameters.current_ep),...
         ', training loss: ', num2str(opts.results.TrainEpochLoss(end)),...
                ', top 1 err: ', num2str(opts.results.TrainEpochError(end)),...
                ',top 5 err:,',num2str(opts.results.TrainEpochError_Top5(end))])                

    end

    
    if opts.RecordStats==1
        if ~isfield(opts,'results')||~isfield(opts.results,'TrainMiniBatchLoss')
            opts.results.TrainMiniBatchLoss=[];
            opts.results.TrainMiniBatchError=[];
            opts.results.TrainMiniBatchError_Top5=[];
        end
        opts.results.TrainMiniBatchLoss=[opts.results.TrainLoss;opts.TrainMiniBatchLoss];
        opts.results.TrainMiniBatchError=[opts.results.TrainError;opts.TrainMiniBatchError]; 
        opts.results.TrainMiniBatchError_Top5=[opts.results.TrainError_Top5;opts.TrainMiniBatchError_Top5]; 
        
    end
    
    toc;

end




