function [  net,res,opts ] = sgd_bit(  net,res,opts )
%Stochastic gradient descent algorithm


    if ~isfield(opts.parameters,'second_order')
        opts.parameters.second_order=0;
    end
    if opts.parameters.second_order
        [  net,res,opts ] = gradient_decorrelation(  net,res,opts );
    end
    
    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=1e-4;
    end
    
    if ~isfield(opts.parameters,'clip')
        opts.parameters.clip=0;
    end
    
    if ~isfield(net,'iterations')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
        net.iterations=0;
    end

    
    net.iterations=net.iterations+1;    
    mom_factor=(1-opts.parameters.mom.^net.iterations);
    
    
    
    for layer=1:numel(net.layers)
         if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)
%           switch net.layers{layer}.type
% 
%             case {'conv' , 'conv2d'}
            if opts.parameters.clip>0
                mask=abs(res(layer).dzdw)>opts.parameters.clip;
                res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
                mask=abs(res(layer).dzdb)>opts.parameters.clip;
                res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
            end
            
           C=eq(net.layers{1, layer}.weights{1, 7},res(layer).dzdw);
           mask=logical(C-1);
           net.layers{1, layer}.weights{1, 6}=net.layers{1, layer}.weights{1, 6}.*mask;
        
%             if opts.parameters.clip>0
%                 mask=abs(res(layer).dzdw)>opts.parameters.clip;
%                 res(layer).dzdw(mask)=sign(res(layer).dzdw(mask)).*opts.parameters.clip;%%this type of processing seems to be very helpful
%                 mask=abs(res(layer).dzdb)>opts.parameters.clip;
%                 res(layer).dzdb(mask)=sign(res(layer).dzdb(mask)).*opts.parameters.clip;
%             end
%          if any(layer~=1)
%           res(layer).dzdw=res(layer).dzdw/1.0e+07;
%           res(layer).dzdb=res(layer).dzdb/1000;
%           end
            %% method 1
            
%             if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
%                 net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
%                 net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
%                 
%             end
%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.l1_lr_1*net.layers{layer}.momentum{1}./mom_factor;

%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*res(layer).dzdb;
             %% method 3
%           net.layers{1,layer}.weights{1}=net.layers{1,layer}.weights{1}-opts.parameters.l1_lr_1*(res(layer).dzdw+net.layers{1, layer}.weights{1, 7})+opts.parameters.l1_lr_3*net.layers{1, layer}.weights{1, 6};
%           net.layers{1,layer}.weights{2}=net.layers{1,layer}.weights{2}-opts.parameters.l1_lr_1*(res(layer).dzdb+net.layers{1, layer}.weights{1, 9})+opts.parameters.l1_lr_3*net.layers{1, layer}.weights{1, 8};

%             if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
%             net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
%             net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});              
%             end
%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.l1_lr_1*net.layers{layer}.momentum{1}./mom_factor;
%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*res(layer).dzdb;

%          %% method 2
         net.layers{1,layer}.weights{1}=net.layers{layer}.weights{1}-opts.parameters.l1_lr_2*(res(layer).dzdw+net.layers{1, layer}.weights{1, 7}-net.layers{1, layer}.weights{1, 6});
         net.layers{1,layer}.weights{2}=net.layers{layer}.weights{2}-opts.parameters.l1_lr_2*(res(layer).dzdb+net.layers{1, layer}.weights{1, 9}-net.layers{1, layer}.weights{1, 8});   
         net.layers{1, layer}.weights{1, 1}=net.layers{1, layer}.weights{1, 1}.*net.layers{1, layer}.weights{1, 3};
%          net.layers{1,layer}.weights{1}=max(-1,min(1,net.layers{1,layer}.weights{1}));
%          net.layers{1,layer}.weights{2}=max(-1,min(1,net.layers{1,layer}.weights{2}));
         net.layers{1, layer}.weights{1,6}=(1/opts.n_train)*(net.layers{1, layer}.weights{1,6}+res(layer).dzdw);
         net.layers{1, layer}.weights{1,7}=res(layer).dzdw;
         net.layers{1, layer}.weights{1,8}=(1/opts.n_train)*(net.layers{1, layer}.weights{1,8}+res(layer).dzdb);
         net.layers{1, layer}.weights{1,9}=res(layer).dzdb;
%          net.layers{1,layer}.weights{6}=max(-1,min(1,net.layers{1,layer}.weights{6}));
%          net.layers{1,layer}.weights{7}=max(-1,min(1,net.layers{1,layer}.weights{7}));
%          net.layers{1,layer}.weights{8}=max(-1,min(1,net.layers{1,layer}.weights{8}));
%          net.layers{1,layer}.weights{9}=max(-1,min(1,net.layers{1,layer}.weights{9}));
%          if any(layer~=1) || any(layer~=12) || any(layer~=4)
%           net.layers{1,layer}.weights{1}=max(-1,min(1,net.layers{1,layer}.weights{1}))   ;
%          end
              %% method 1 
% %             
%             if ~isfield(net.layers{layer},'momentum')||(isfield(opts,'reset_mom')&&opts.reset_mom==1)
%                 net.layers{layer}.momentum{1}=zeros(size(net.layers{layer}.weights{1}),'like',net.layers{layer}.weights{1});
%                 net.layers{layer}.momentum{2}=zeros(size(net.layers{layer}.weights{2}),'like',net.layers{layer}.weights{2});
%                 
%             end
%             net.layers{layer}.momentum{1}=opts.parameters.mom.*net.layers{layer}.momentum{1}-(1-opts.parameters.mom).*res(layer).dzdw - opts.parameters.weightDecay * net.layers{layer}.weights{1};
%             net.layers{layer}.weights{1}=net.layers{layer}.weights{1}+opts.parameters.l1_lr_1*net.layers{layer}.momentum{1}./mom_factor;
%             
%             net.layers{layer}.momentum{2}=opts.parameters.mom.*net.layers{layer}.momentum{2}-(1-opts.parameters.mom).*res(layer).dzdb;
%             net.layers{layer}.weights{2}=net.layers{layer}.weights{2}+opts.parameters.l1_lr_1*net.layers{layer}.momentum{2}./mom_factor;
% %            

        %% clipping the weights using L1 regularization 
% % % %         net.layers{1,layer}.weights{1, 5}=net.layers{1,layer}.weights{1, 1};
% % % %         opts.weight= net.layers{1,1}.weights{1, 1};
% % % %         weight=net.layers{1,layer}.weights{1, 1};
% % % %         pos_mask=weight>0;
% % % %         pos_weight=weight.*pos_mask;
% % % %         neg_mask=weight<0;
% % % %         neg_weight=weight.*neg_mask;
% % % % %         %% method1
% % % % % % %         pos_mask_up=pos_mask.*opts.parameters.v;
% % % % % % %         neg_mask_up=neg_mask.*opts.parameters.v;
% % % % %        %% method2
% % % %         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4};
% % % %         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4};
% % % %          %% method3
% % % % %         pos_bias_mask=net.layers{1,layer}.weights{1, 2}>0;
% % % % %         neg_bias_mask=net.layers{1,layer}.weights{1, 2}<0;
% % % % %         pos_bias=pos_bias_mask.*net.layers{1,layer}.weights{1, 2};      
% % % % %         min_pos_bias=pos_bias(find(pos_bias>0));
% % % % %         opts.ppos_bias=min(min_pos_bias);
% % % % %         neg_bias=neg_bias_mask.*net.layers{1,layer}.weights{1, 2};   
% % % % %         min_neg_bias=neg_bias(find(neg_bias<0));
% % % % %         opts.nneg_bias=min(min_neg_bias); 
% % % % %         if isempty(opts.ppos_bias) 
% % % % %             opts.ppos_bias=0;
% % % % %         end
% % % % %         if isempty(opts.nneg_bias)
% % % % %             opts.nneg_bias=0;
% % % % %         end
% % % % % 
% % % % %         pos_mask_up=opts.parameters.u+net.layers{1,layer}.weights{1, 4}+opts.ppos_bias;
% % % % %         neg_mask_up=opts.parameters.u-net.layers{1,layer}.weights{1, 4}-opts.nneg_bias;
% % % % % % %         %% l1 norm
% % % %         pos_weight_up=max(0,pos_weight-pos_mask_up);
% % % %         neg_weight_up=min(0,neg_weight+neg_mask_up);
% % % %         net.layers{1,layer}.weights{1, 1}=pos_weight_up+neg_weight_up;
% % % % % 
% % % %         net.layers{1, layer}.weights{1, 4}=net.layers{1, layer}.weights{1, 4}+(net.layers{1, layer}.weights{1, 1}-net.layers{1,layer}.weights{1, 5});       

         end
    end
   
    if ~isfield(opts,'reset_mom')||opts.reset_mom==1
        opts.reset_mom=0;
    end
end

