function [  net,res,opts ] = sgd2_SVRG(  net,res,opts )
% Stochastic gradient descent using second-order information.
%   Ye, C., Yang, Y., Fermuller, C., & Aloimonos, Y. (2017). 
%   On the Importance of Consistency in Training Deep Neural Networks. arXiv preprint arXiv:1708.00631.
    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end    
    if ~isfield(opts.parameters,'lambda_sgd2')
        opts.parameters.lambda_sgd2=1e0;
    end
    if ~isfield(opts.parameters,'large_matrix_inversion')
        opts.parameters.large_matrix_inversion=0;
    end
    if ~isfield(opts.parameters,'max_inv_size')
        opts.parameters.max_inv_size=500;
    end    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=0;
    end
    if ~isfield(opts.parameters,'decorr_bias')
        opts.parameters.decorr_bias=1;
    end
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end
   
    net.iterations=net.iterations+1;
    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    max_inv_size=opts.parameters.max_inv_size;
    lambda=opts.parameters.lambda_sgd2;
    
    
    for layer=1:numel(net.layers)
        if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
            if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
            end
            
            dzdw=res(layer).dzdw;
            dzdb=res(layer).dzdb;
            
%              if length(net.layers{layer}.weights)==2
                x=res(layer).x;
                batch_dim=length(size(x));%This assumes the batch size must be >1 
                if batch_dim==4%2d cnn
                    x=permute(x,[3,1,2,4]);x=reshape(x,size(x,1),[]);
                    dzdw=permute(dzdw,[1,2,4,3]);
                    new_size=size(dzdw);
                    dzdw=reshape(dzdw,prod(new_size(1:3)),new_size(4));
                    K=size(dzdw,1)/numel(dzdb);dzdb=repelem(dzdb(:),K,1);
                end
                if batch_dim==3%1d cnn
                    x=permute(x,[2,1,3]);x=reshape(x,size(x,1),[]);
                    dzdw=permute(dzdw,[1,3,2]);
                    new_size=size(dzdw);
                    dzdw=reshape(dzdw,prod(new_size(1:2)),new_size(3));
                    K=size(dzdw,1)/numel(dzdb);dzdb=repelem(dzdb(:),K,1);
                end
                subsample=1;batch_size=size(x,2);
                if batch_size>1e4,subsample=ceil(min(50,batch_size/1e4));end
                if subsample>1,x=x(:,1:subsample:end);end
                if opts.parameters.decorr_bias==1
                    %insert bias
                    x=[ones(1,size(x,2),'like',x);x];
                    dzdw=[dzdb,dzdw];
                end
                if size(dzdw,2)<=max_inv_size %small scale inversion
                    dzdw=dzdw/(x*x'./size(x,2)+lambda*eye(size(x,1),'like',x));  
                elseif opts.parameters.large_matrix_inversion %divide large scale into smaller scale
                    order=randperm(size(dzdw,2));
                    for i=1:max_inv_size:length(order) %could have been parallelized 
                       block_size=min(max_inv_size,length(order)-i+1);
                       idx=order(i:i+block_size-1);x_tmp=x(idx,:);
                       dzdw(:,idx)=(dzdw(:,idx)/(x_tmp*x_tmp'./size(x_tmp,2)+lambda*eye(size(x_tmp,1),'like',x)));
                    end
                end
                if opts.parameters.decorr_bias==1
                    dzdb=dzdw(:,1);dzdw(:,1)=[];
                end
                if batch_dim==4,dzdw=reshape(dzdw,new_size);dzdw=permute(dzdw,[1,2,4,3]);end
                if batch_dim==3,dzdw=reshape(dzdw,new_size);dzdw=permute(dzdw,[1,3,2]);end
                if batch_dim>2%for cnn:                      
                    %dzdb is decorrelated with dzdw, take average to smooth the results.                    
                    dzdb=reshape(mean(reshape(dzdb(:),K,[]),1),size(res(layer).dzdb));
                end
%              end
            
            if opts.parameters.clip>0
                mask=abs(res(layer).dzdw)>opts.parameters.clip;
                res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
                mask=abs(res(layer).dzdb)>opts.parameters.clip;
                res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
            end
%           
%          
             if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
                net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
             end
          opts.grad1=opts.grad1+1;  
   
%           net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*dzdb;
%           net.layers{layer}.weights{2}=net.layers{layer}.weights{2}+opts.parameters.lr*net.layers{layer}.momentum{2}./mom_factor;% % 
% 
%           net.layers{layer}.weights{2}=zeros(size(net.layers{layer}.weights{2}));
         

         
         
          %           res(layer).dzdw=(1/opts.parameters.batch_size)*(1-opts.parameters.l1_lr_3)*net.layers{1, layer}.weights{1, 3}+(opts.parameters.l1_lr_3)*net.layers{1, layer}.weights{1, 6};
%         if opts.ep==1
%             net.layers{1, layer}.weights{1, 6}=(1/opts.mini_b)*(net.layers{1, layer}.weights{1, 6}+dzdw);
%         end
        weight_un=net.layers{layer}.weights{1};
%         mask=(net.layers{layer}.weights{1}~=0);

%        Variance=normalize(variance,'center','median');
%         Dzdw=normalize(Dzdw1,'center','median'); 

       variance=(1-opts.parameters.l1_lr_3)*net.layers{1, layer}.weights{1,3}-opts.parameters.l1_lr_3*(net.layers{1, layer}.weights{1, 6});
%       if  opts.parameters.l1_lr_3==0
%        res(layer).dzdw=  dzdw; 
       res(layer).dzdw=(1-opts.parameters.l1_lr_3)*(dzdw)-variance;% 
        net.layers{1, layer}.weights{1,5}=variance;
%        net.layers{1, layer}.weights{1,5}=(net.layers{1, layer}.weights{1,3}-net.layers{1, layer}.weights{1, 6});
%        net.layers{1, layer}.weights{1,3}+(net.layers{1, layer}.weights{1, 6});
%        %% 
%        if opts.ep<=1
%        opts.Dzdw=((1-opts.parameters.l1_lr_3)*dzdw-(opts.parameters.l1_lr_3)*net.layers{1, layer}.weights{1, 6}).^(2); 
%        else
  
%        opts.Dzdw=gather(max(max(opts.Dzdw,[],4),[],3));
       %           net.layers{layer}.weights{2}=net.layers{layer}.weights{2}-0.2* res(layer).dzdb;
%          ffx=net.layers{1, 1}.weights{1, 6};
%           FFx=sum(ffx.^(2),2);
%           FFX=sqrt(FFx);
%           opts.fx(1, opts.ep)=FFX(2);   
%                           net.layers{1, layer}.weights{1,10}=net.layers{1, layer}.weights{1,10}-res(layer).dzdw;
%                           net.layers{1,layer}.weights{1, 6}=(1/opts.n_train)*net.layers{1, layer}.weights{1,10};
  
%      [aa,bb]=min(opts.TrainMiniBatchLoss);
%          if opts.mini_b==bb
          net.layers{layer}.weights{1}=net.layers{layer}.weights{1}-(opts.parameters.l1_lr_2* res(layer).dzdw);
          net.layers{layer}.weights{2}=net.layers{layer}.weights{2}-0.09*res(layer).dzdb;
%          end

         weight1=net.layers{1}.weights{1};
         Weight=sqrt(sum(weight1.^(2),3));
         opts.weight(1,opts.ep)=Weight(3);
          %           weight=net.layers{1,layer}.weights{1, 1};
%           opts.weight(opts.grad1)=weight(1,1,1,1);
%           opts.gradient(opts.grad1)=res(layer).dzdw(1,1,1,1);
%          net.layers{1, layer}.weights{1, 6}=(1/opts.parameters.batch_size)*(net.layers{1, layer}.weights{1, 6}+net.layers{1, layer}.weights{1, 7});
% 
          
          %          res(layer).dzdw=Dzdw; 

%           fx=opts.parameters.l1_lr_2*res(1).dzdw;
%          if fx(5,1,1,1)<0
%            fx(5,1,1,1)=-fx(5,1,1,1);
%          else
%            fx(5,1,1,1)=fx(5,1,1,1);
%          end
%          opts.fx(1,opts.grad1)=(fx(5,1,1,1));   

% % %         if fx(1,11)<0
% % %            fx(1,11)=-fx(1,11);
% % %          else
% % %            fx(1,11)=fx(1,11);
% % %          end
% % %          opts.fx(1,opts.grad1)=(fx(1,11));   

%        net.layers{1, layer}.weights{1,10}=(net.layers{1, layer}.weights{1,10}+res(layer).dzdw);    
        
              %% method 1 
%              if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
%                 net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
%                 net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
%             end
% %     
%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.lr*net.layers{layer}.momentum{1}./mom_factor;
% %             
%             net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*dzdb;
%             net.layers{layer}.weights{2}=net.layers{layer}.weights{2}+opts.parameters.lr*net.layers{layer}.momentum{2}./mom_factor;% %            
 
        %% clipping the weights using L1 regularization 
% % %         if opts.myi>1
% % %         net.layers{1,layer}.weights{1, 5}=net.layers{1,layer}.weights{1, 1};
% % %         weight_bias=net.layers{layer}.weights{1};
% % %         dzdw_1=(1-opts.parameters.l1_lr_4)*(dzdw)-(opts.parameters.l1_lr_4)*(net.layers{1, layer}.weights{1, 3}-net.layers{1, layer}.weights{1, 6});
% % %         weight_un=weight_un-mask.*(opts.parameters.l1_lr_2*dzdw_1);
% % %         aa=abs(weight_bias)-abs(weight_un);
% % %         bb=nnz((aa<0));
% % %         cc=nnz((aa>0));        
% % %         if cc>bb
% % %             weight=weight_un;
% % %             res(layer).dzdw=dzdw_1;
% % %         else
% % %             weight=weight_bias;
% % %         end
% % % %         weight=net.layers{1,layer}.weights{1, 1};
% % %         pos_mask=weight>0;
% % %         pos_weight=weight.*pos_mask;
% % %         neg_mask=weight<0;
% % %         neg_weight=weight.*neg_mask;
% % % %         %% method1
% % % % % %         pos_mask_up=pos_mask.*opts.parameters.v;
% % % % % %         neg_mask_up=neg_mask.*opts.parameters.v;
% % % %        %% method2
% % %         pos_mask_up=pos_mask.*(opts.parameters.u+net.layers{1,layer}.weights{1, 4});
% % %         neg_mask_up=neg_mask.*(opts.parameters.u-net.layers{1,layer}.weights{1, 4});
% % %          %% method3          
% % % %         if opts.bias>=0
% % % %         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4}+opts.bias;
% % % %         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4};
% % % %         elseif opts.bias<0
% % % %         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4};    
% % % %         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4}-opts.bias;
% % % %         end
% % %         
% % % % %         %% l1 norm
% % %         pos_weight_up=max(0,pos_weight-pos_mask_up);
% % %         neg_weight_up=min(0,neg_weight+neg_mask_up);
% % %         net.layers{1,layer}.weights{1, 1}=pos_weight_up+neg_weight_up;      
% % %         net.layers{1, layer}.weights{1, 4}=net.layers{1, layer}.weights{1, 4}+(net.layers{1, layer}.weights{1, 1}-net.layers{1,layer}.weights{1, 5});
% % %         end
        if opts.myi>1
        net.layers{1,layer}.weights{1, 5}=net.layers{1,layer}.weights{1, 1};
%        opts.weight= net.layers{1,1}.weights{1, 1};
        weight=net.layers{1,layer}.weights{1, 1};
        pos_mask=weight>0;
        pos_weight=weight.*pos_mask;
        neg_mask=weight<0;
        neg_weight=weight.*neg_mask;
%         %% method1
% % %         pos_mask_up=pos_mask.*opts.parameters.v;
% % %         neg_mask_up=neg_mask.*opts.parameters.v;
%        %% method2
        pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4};
        neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4};
         %% method3          
%         if opts.bias>=0
%         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4}+opts.bias;
%         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4};
%         elseif opts.bias<0
%         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4};    
%         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4}-opts.bias;
%         end
        
% %         %% l1 norm
        pos_weight_up=max(0,pos_weight-pos_mask_up);
        neg_weight_up=min(0,neg_weight+neg_mask_up);
        net.layers{1,layer}.weights{1, 1}=pos_weight_up+neg_weight_up;      
        net.layers{1, layer}.weights{1, 4}=net.layers{1, layer}.weights{1, 4}+(net.layers{1, layer}.weights{1, 1}-net.layers{1,layer}.weights{1, 5});
%         net.layers{1, 1}.weights{1,11}=opts.parameters.l1_lr_2*res(1).dzdw;
        end
        end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end

end

