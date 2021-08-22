function [net,opts]=train_net(net,opts)

    if ~isfield(opts,'datatype')
        opts.datatype='single';
    end
    
    opts.training=1;

    if ~isfield(opts.parameters,'learning_method')
        opts.parameters.learning_method='sgd';        
    end
    
    if ~isfield(opts,'display_msg')
        opts.display_msg=1; 
    end
    opts.TrainMiniBatchError=[];
    opts.TrainMiniBatchError_Top5=[];
    opts.TrainMiniBatchLoss=[];
    
    
    tic
    
    tstart=tic;

    if opts.parameters.selective_sgd==1 
        [ net,opts ] = selective_sgd( net,opts );
    end
    
%     opts.batchsize=randperm(opts.n_train);
%     if opts.parameters.l1_lr_2==opts.l1_lr_1_b
%     opts.order_batch=floor((2*opts.n_train^(2))/(2*opts.n_train+(1-opts.parameters.l1_lr_3)*opts.n_train^(3/2)*0.01));
%     opts.order=opts.batchsize(randperm(opts.order_batch));
%  %   opts.order=opts.batchsize(1:opts.order_batch);
%     else
%     opts.order_batch=floor((2*opts.n_train^(2))/(2*opts.n_train+(opts.parameters.l1_lr_3)*opts.n_train^(3/2)*0.01));   
%     opts.order=opts.batchsize(randperm(opts.order_batch));
% %    opts.order=opts.batchsize(1:opts.order_batch);
%     end
%     opts.leftorder=setxor(opts.order,opts.batchsize);
%     
%     if length(size(opts.train))>=3
%     opts.train_batch=opts.train(:,:,:,opts.order);
%     opts.train_batch_left=opts.train(:,:,:, opts.leftorder);
%     else
%      opts.train_batch=opts.train(:,opts.order);  
%      opts.train_batch_left=opts.train(:,opts.leftorder);
%     end
%     idx_outer=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);

    
    batch_dim=length(size(opts.train));
    idx_nd=repmat({':'},[1,batch_dim]);
%     nn1=1/(1+opts.ep/opts.n_epoch);
%     opts.parameters.l1_lr_1=opts.sgd_lr_1*nn1;
    nn2=opts.ep/opts.n_epoch;
    opts.parameters.l1_lr_4=opts.sgd_lr_3*opts.alpha^(nn2);
%     nn3=1/(1+0.5*(opts.ep/opts.n_epoch)^(3));
%     opts.parameters.l1_lr_3=opts.sgd_lr_3*nn3; 
    opts.parameters.u=opts.parameters.u+opts.parameters.l1_lr_4*(opts.parameters.lamda/opts.n_train);%
%     nn2=(opts.ep*opts.n_train_batch*opts.parameters.batch_size)/opts.n_train;    
% %     opts.lr4=opts.lr4_0*nn4;  
% %     nn2=opts.ep/opts.parameters.batch_size;
%     opts.parameters.l1_lr_4=opts.sgd_lr_1*opts.alpha^(nn2);
%     opts.parameters.u=opts.parameters.u+opts.parameters.l1_lr_4*(opts.parameters.lamda/opts.n_train);%
% %        for layer=1:numel(net.layers)
%         if isfield(net.layers{1,layer},'weights')
% net.layers{1, layer}.weights{1, 7}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%         end
%        end

   for mini_b=1:opts.n_train_batch

       opts.mini_b=mini_b;
       %       opts.bound=  (0.4*opts.parameters.batch_size);
       if (~isfield(opts.parameters,'iterations'))
            opts.parameters.iterations=0; 
        end  
%     nn3=1/(1+(mini_b/opts.parameters.batch_size));   1+(opts.ep^(0.25))/10
%     opts.parameters.l1_lr_3=opts.sgd_lr_3*nn3;
 
%    idx=datasample(opts.batchsize,opts.parameters.batch_size); 
%       idx=opts.idx_initial;
%         idx_sgd=opts.leftorder(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);
       
%         signal=ismember(opts.ep,opts.xrange);       
%        if signal==1   
%        idx_total=opts.batchsize(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);    
% %          if numel(opts.c)> opts.bound_b || numel(opts.c)> opts.bound_a
%         idx_nd{end}=idx_total;
%         opts.idx_nd=idx_nd;
%         opts.idx=idx_total;
%         res(1).x=opts.train(idx_nd{:}); 
%         if strcmp(net.layers{end}.type,'softmaxloss')
%             res(1).class=opts.train_labels(idx_total);
%         end
%        else
%  signal=ismember(opts.ep,opts.xrange);       
%   if signal==1
%         idx_total=opts.batchsize(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);         
%        idx_nd{end}=idx_total;
%         opts.idx_nd=idx_nd;
%         opts.idx=idx_total;
%         res(1).x=opts.train(idx_nd{:}); 
%         if strcmp(net.layers{end}.type,'softmaxloss')
%             res(1).class=opts.train_labels(idx_total);
%         end
%   else  
       idx_total=opts.batchsize(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);         
        opts.C_defin=ismember(idx_total,opts.order);  
       [opts.c,opts.idx_order]=find(opts.C_defin==0); 
     if mini_b==1
         size_c=numel(opts.c);   
    opts.bound_b= floor(1.1*(size_c));
    opts.bound_a=floor(1*(size_c));
     end
        idx=opts.order(1+(mini_b-1)*opts.parameters.batch_size:mini_b*opts.parameters.batch_size);   
         
%        if numel(opts.c)<= opts.bound_b || numel(opts.c)<= opts.bound_a
%         idx_nd{end}=idx;
%         opts.idx_nd=idx_nd;
%         opts.idx=idx;
%         res(1).x=opts.train(idx_nd{:}); 
%             if strcmp(net.layers{end}.type,'softmaxloss')
%             res(1).class=opts.train_labels(idx);
%             end
%        else
        idx_nd{end}=idx;
        opts.idx_nd=idx_nd;
        opts.idx=idx;
        res(1).x=opts.train(idx_nd{:}); 
            if strcmp(net.layers{end}.type,'softmaxloss')
            res(1).class=opts.train_labels(idx);
            end
%        end
%   end
        %classification
    
        
%          signal=ismember(opts.ep,opts.xrange);
%           if signal==1
%               if mini_b==opts.n_train_batch
%           opts.res=res; 
%               end
%           end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%forward%%%%%%%%%%%%%%%%%%%
        [ net,res,opts ] = net_ff( net,res,opts );    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%backward%%%%%%%%%%%%%%%%
        opts.dzdy=1.0;
        
        [ net,res,opts ] = net_bp( net,res,opts );
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
% % %          for layer=1:numel(net.layers)
% % %         if isfield(net.layers{layer},'weights')
% % %             if opts.ep==1
% % %                  Dzdw3=normalize(res(layer).dzdw,'center','mean');
% % %                  net.layers{1,layer}.weights{1, 6}=Dzdw3;
% % %             end
% % %         end
% % %          end

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
%         Proxsgd

%        proxsvrg
%  signal=ismember(opts.ep,opts.xrange);       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%stochastic gradients descent%%%%%%%%%%%%%%%%%%%%%%%%
%         [c,idx_order]=find(opts.C_defin==0);
% % %          if opts.parameters.l1_lr_2==opts.l1_lr_1_a  
         if  opts.parameters.l1_lr_3<=0.5
%         if  opts.parameters.l1_lr_3>=0.67
%             [ net,res,opts ] = sgd2_SVRG( net,res,opts );
%         else  
%            signal=ismember(opts.ep,opts.xrange);       
%           if signal==1    
%           opts.parameters.l1_lr_3=1; 
%             else
%           if isempty(opts.idx_order) 
% % % % %              if  numel(opts.c)>opts.bound_a  
% % % % %             opts.parameters.l1_lr_3=1; 
% % % % %              end
%           end
%            end
          [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts ); 
%          elseif opts.order_batch==opts.order_batch_c
%              [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts ); 
%              opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(0.65*opts.Lamdba);
%           [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts );
%           end
%         end
% % % %         elseif opts.parameters.l1_lr_2==opts.l1_lr_1_b
         elseif  opts.parameters.l1_lr_3>0.5
%            if opts.ep==1
%                opts.parameters.l1_lr_3=0.8;    
%            [ net,res,opts ] = sgd2_SVRG( net,res,opts ); 
%            else       
%             if signal==1 && numel(opts.c)>opts.bound_b
%           if mod(opts.ep,opts.q)==0
%          if opts.ep==1
%             opts.parameters.l1_lr_3=1;    
%           [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts ); 
%          else
%           if signal==1          
%               opts.parameters.l1_lr_3=1;    
%           [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts ); 
%           else
% % %              if numel(opts.c)>opts.bound_b
% % %            opts.parameters.l1_lr_3=1;    
% % %           [ net,res,opts ] = sgd2_SVRG_unb( net,res,opts );       
% % %             else    
            [ net,res,opts ] =sgd2_SVRG( net,res,opts ); 
% % %              end
%            end
%          end
         end
       for layer=1:numel(net.layers)
         if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
%        end
         signal=ismember(opts.ep,opts.xrange);           
          if signal==1
        nn=size(net.layers{1, layer}.weights{1, 5});
        if numel(nn)<=2
            N_size=nn(1)*nn(2);
        else
            N_size=nn(1)*nn(2)*nn(3)*nn(4);
        end
%        net.layers{1, layer}.weights{1,5}=normalize(net.layers{1, layer}.weights{1,5},'center','median');
       opts.Dzdw1=(sum(reshape(net.layers{1, layer}.weights{1,5},[1 N_size]),2));
%             opts.Dzdw1=sum(sum(sum(sum(net.layers{1, layer}.weights{1, 5},2),1),3),4);
            opts.Dzdw=abs(gather(opts.Dzdw1^(2)));             
        net.layers{1, layer}.weights{1,12}= (net.layers{1, layer}.weights{1,12}+res(layer).dzdw);
%         res(layer).Dzdw=res(layer).dzdw;  
%        opts.rand=randperm(floor(opts.n_train_batch),1);

%        if mini_b==opts.rand
%       [aa,bb]=min(opts.TrainMiniBatchLoss);
%           if mini_b==bb 
       net.layers{1, layer}.weights{1,10}=res(layer).dzdw;
%           end
   
%             if opts.ep==1
%             opts.Dzdw=0;
%             else
%            normvariance=normalize(gather(net.layers{1, layer}.weights{1,5}),'norm');
% % % % %             nn=size(net.layers{1, layer}.weights{1, 5});
% % % % %             N_size=nn(1)*nn(2)*nn(3)*nn(4);
% % % % %              opts.Dzdw1=(sum(reshape(net.layers{1, layer}.weights{1,5},[1 N_size]),2));
% % % % % %             opts.Dzdw1=sum(sum(sum(sum(net.layers{1, layer}.weights{1, 5},2),1),3),4);
% % % % %             opts.Dzdw=abs(gather(opts.Dzdw1)); 
%           net.layers{1, layer}.weights{1, 5}=net.layers{1, layer}.weights{1, 5}+((net.layers{1, layer}.weights{1, 10}-net.layers{1, layer}.weights{1, 6}).^(2));
%             end
           end
         end
       end
 
     
        %           end
%          net.layers{1,layer}.weights{1, 13}=(net.layers{1,layer}.weights{1,14}-res(layer).dzdw); 
%          net.layers{1,layer}.weights{1, 6}=net.layers{1,layer}.weights{1, 6}-(1/opts.n_train_batch)*net.layers{1,layer}.weights{1, 13};
%           net.layers{1,layer}.weights{1, 3}=net.layers{1, layer}.weights{1,10};
        
 
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%               if opts.mini_b==opts.n_train_batch
%           net.layers{1, layer}.weights{1,3}=opts.res(layer).dzdw; 
%    end  
     opts.count=opts.count+1 ;

  end
     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for layer=1:numel(net.layers)
      if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
   
   
%             opts.Dzdw=sqrt(opts.Dzdw);   
                signal=ismember(opts.ep,opts.xrange);
               
      if signal==1 
      net.layers{1,layer}.weights{1, 6}=(1/opts.n_train_batch)*net.layers{1, layer}.weights{1,12};            
%       if opts.order_batch==opts.order_batch_a
%         net.layers{1, layer}.weights{1,3}=((opts.parameters.l1_lr_3)*(res(layer).Dzdw-net.layers{1, layer}.weights{1, 6}));
%       else
%        net.layers{1, layer}.weights{1,3}=(1-opts.parameters.l1_lr_3)*res(layer).Dzdw-(opts.parameters.l1_lr_3)*(net.layers{1, layer}.weights{1, 6});
%       end          
      net.layers{1, layer}.weights{1,3}=net.layers{1, layer}.weights{1,10} ; 
          
%             net.layers{1, layer}.weights{1,3}=net.layers{1, layer}.weights{1,10};                
%                     if  opts.parameters.l1_lr_3>0.65
%                           net.layers{1, layer}.weights{1,10}=net.layers{1, layer}.weights{1,10}+res(layer).dzdw;
          
%           if opts.parameters.l1_lr_2==opts.l1_lr_1_b opts.order_batch
%          net.layers{1, layer}.weights{1,3}=res(layer).dzdw;
%           elseif opts.parameters.l1_lr_2==opts.l1_lr_1_a
          
     end 
           net.layers{1, layer}.weights{1, 12}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');

%                end
         
%             net.layers{1, layer}.weights{1,6}=res(layer).dzdw;
         
%  
%          net.layers{1, layer}.weights{1,6}= net.layers{1, layer}.weights{1,6}-(1/opts.n_train_batch)*( net.layers{1, layer}.weights{1,6}-net.layers{1, layer}.weights{1,3});
%          net.layers{1, layer}.weights{1, 13}=res(layer).dzdw;  
%                         signal=ismember(opts.ep,opts.xrange);
%           if signal==1  
%            net.layers{1, layer}.weights{1,3}=net.layers{1, layer}.weights{1,10};
%                     if  opts.parameters.l1_lr_3>0.65
%                           net.layers{1, layer}.weights{1,10}=net.layers{1, layer}.weights{1,10}+res(layer).dzdw;
%                           net.layers{1,layer}.weights{1, 6}=(1/opts.n_train)*net.layers{1, layer}.weights{1,10};
%           if opts.parameters.l1_lr_2==opts.l1_lr_1_b 
%          net.layers{1, layer}.weights{1,3}=res(layer).dzdw;
%           elseif opts.parameters.l1_lr_2==opts.l1_lr_1_a
          
%           end
          %end
 
%          end
%            if layer==1
%           ffx=res(layer).dzdw;
%            FFx2=sqrt(sum(ffx.^(2),3));
%              opts.fx1(1, opts.ep)=FFx2(3);           
%            end
%           end
        
%           if  opts.parameters.l1_lr_3<=0.65
          opts.Weight{1,opts.ep}=net.layers{1, layer}.weights{1,1}; 
% %             if layer==1
% %           ffx=res(layer).dzdw;
% %            FFx2=sqrt(sum(ffx.^(2),3));
% % 
% % %            ffx2=res(layer).dzdw;
% % %            FFx2=sqrt(sum(ffx2.^(2),3));
% % % 
% %            opts.fx(1, opts.ep)=FFx2(3);           
% %             end

%            net.layers{1,layer}.weights{1, 12}=(net.layers{1, layer}.weights{1,12}+net.layers{1, layer}.weights{1,3}); %
%        net.layers{1, layer}.weights{1,6}= (1/(opts.n_epoch))*(net.layers{1, layer}.weights{1, 3});
 
%           elseif opts.parameters.l1_lr_3>0.65
%           
%           net.layers{1, layer}.weights{1, 6}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%           net.layers{1, layer}.weights{1, 10}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%           end

          if layer==1
          ffx=res(layer).dzdw;
           FFx2=sqrt(sum(abs(ffx(12)),3));%cnn
%           FFx2=sqrt(sum(abs(ffx(12,1:end)),2));%mlp 
           opts.fx(1, opts.ep)=FFx2;           
         end
      end
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
  opts.tElapsed(1,opts.ep)=toc(tstart);
   end




