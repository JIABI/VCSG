function [opts]=test_net(net,opts)

    opts.training=0;

    opts.TestMiniBatchError=[];
    opts.TestMiniBatchError_Top5=[];
    opts.TestMiniBatchLoss=[];
  
    
    if ~isfield(opts,'validating')
        opts.validating=0;
    end
    
    if opts.validating
        n_batch=opts.n_valid_batch;
    else
        n_batch=(opts.n_test/opts.parameters.batch_size);
    end

       
%     end
% % %   if opts.parameters.l1_lr_2==opts.l1_lr_1_b
% % %     opts.order_batch_test=round((2*opts.n_test^(2))/(2*opts.n_test+(1-0.8)*opts.n_test^(3/2)*0.1));
% % %  %   opts.order=randperm(opts.order_batch);
% % %  %   opts.order=opts.batchsize(1:opts.order_batch);
% % %     else
% % %     opts.order_batch_test=round((2*opts.n_test^(2))/(2*opts.n_test+(0.1)*opts.n_test^(3/2)*0.009));   
% % %  %   opts.order=randperm(opts.order_batch);
% % % %    opts.order=opts.batchsize(1:opts.order_batch);
% % %   end
%    n_batch=opts.order_batch_test/opts.parameters.batch_size;
% % % %   if size(opts.test)>=3
% % % %   opts.test_batch=opts.test(:,:,:,opts.order_batch_test);
% % % %   else
% % % %   opts.test_batch=opts.test(:,opts.order_batch_test);
% % % %   end   
    batch_dim=length(size(opts.test));
    idx_nd=repmat({':'},[1,batch_dim]);
    
    for mini_b=1:n_batch
        
        idx=1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size;
        idx_nd{end}=idx;
        opts.idx_nd=idx_nd;
        opts.idx=idx;
        
        if opts.validating
            %input
            res(1).x=opts.valid(idx_nd{:});
            %output
            %classification
            if strcmp(net.layers{end}.type,'softmaxloss')
                res(1).class=opts.valid_labels(idx);
            end
        else
            %input
            res(1).x=opts.test(idx_nd{:});
            %output
            %classification
            if strcmp(net.layers{end}.type,'softmaxloss')
                res(1).class=opts.test_labels(idx);
            end
        end
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
%         loss1=double(gather(min(res(end).x(:))));
       loss=double(gather(mean(res(end).x(:))));
        opts.TestMiniBatchLoss=[opts.TestMiniBatchLoss;loss];
        
        if strcmp(net.layers{end}.type,'softmaxloss')
            err=error_multiclass(res(1).class,res);    
            opts.TestMiniBatchError=[opts.TestMiniBatchError;err(1)/opts.parameters.batch_size];
            opts.TestMiniBatchError_Top5=[opts.TestMiniBatchError_Top5;err(2)/opts.parameters.batch_size];
        end
        
       
      
    end

    
    if opts.validating
        if ~isfield(opts,'results')||~isfield(opts.results,'ValidEpochLoss')
            opts.results.ValidEpochLoss=[];
            opts.results.ValidEpochError=[];
            opts.results.ValidEpochError_Top5=[];
          
        end
        opts.results.ValidEpochLoss=[opts.results.ValidEpochLoss;mean(opts.TestMiniBatchLoss(:))];
        if strcmp(net.layers{end}.type,'softmaxloss')
            opts.results.ValidEpochError=[opts.results.ValidEpochError;mean(opts.TestMiniBatchError(:))];
            opts.results.ValidEpochError_Top5=[opts.results.ValidEpochError_Top5;mean(opts.TestMiniBatchError_Top5(:))];
             disp(['Epoch ',num2str(opts.parameters.current_ep),...
                 ', validation loss: ', num2str(opts.results.ValidEpochLoss(end)),...
                        ', top 1 err: ', num2str(opts.results.ValidEpochError(end)),...
                        ',top 5 err:,',num2str(opts.results.ValidEpochError_Top5(end))])                
        end
        
    
    else
        if ~isfield(opts,'results')||~isfield(opts.results,'TestEpochLoss')
        
            opts.results.TestEpochLoss=[];
            opts.results.TestEpochError=[];
            opts.results.TestEpochError_Top5=[];
    
        end        
        opts.results.TestEpochLoss=[opts.results.TestEpochLoss;mean(opts.TestMiniBatchLoss(:))];
        if strcmp(net.layers{end}.type,'softmaxloss')
            opts.results.TestEpochError=[opts.results.TestEpochError;mean(opts.TestMiniBatchError(:))];
            opts.results.TestEpochError_Top5=[opts.results.TestEpochError_Top5;mean(opts.TestMiniBatchError_Top5(:))];
             disp(['Epoch ',num2str(opts.parameters.current_ep),...
                 ', testing loss: ', num2str(opts.results.TestEpochLoss(end)),...
                        ', top 1 err: ', num2str(opts.results.TestEpochError(end)),...
                        ',top 5 err:,',num2str(opts.results.TestEpochError_Top5(end))])                
        end
        
    
    end
    
    
end


