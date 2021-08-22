

files=dir([fullfile(opts.output_dir,opts.saved_filenames)]);

if opts.LoadNet && length(files)>1
    [~,last_file]=sort([files(:).datenum],'descend');
    if length(files)<opts.n_epoch,end  
    load(fullfile(opts.output_dir,files(last_file(1)).name));
    opts.parameters=parameters;
    opts.results=results;
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
end

if opts.LoadNet==0 || length(files)==0      
    net=NetInit(opts);
    opts.results=[];

end

% if opts.LoadNet==1 
%     net=net(opts);
%     opts.results=opts.results;
% 
% end

opts.RecordStats=0;
opts.n_train_batch=floor(opts.n_train/opts.parameters.batch_size);
if isfield(opts,'n_valid')
    opts.n_valid_batch=around(opts.n_valid/opts.parameters.batch_size);
end
opts.n_test_batch=floor(opts.n_test/opts.parameters.batch_size);
if(opts.use_gpu)       
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'gpu');
    end
else
    for i=1:length(net)
        net(i)=SwitchProcessor(net(i),'cpu');
    end
end

start_ep=opts.parameters.current_ep;
if opts.plot
    figure1=figure;
end
for layer=1:numel(net.layers)
        if isfield(net.layers{1,layer},'weights')
net.layers{1, layer}.weights{1, 3}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 5}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 4}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 7}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 6}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 8}=zeros(size(net.layers{1, layer}.weights{1, 2}),'single');
net.layers{1, layer}.weights{1, 9}=zeros(size(net.layers{1, layer}.weights{1, 2}),'single');
net.layers{1, layer}.weights{1, 10}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1,11}=zeros(size(net.layers{1, layer}.weights{1, 2}),'single');
net.layers{1, layer}.weights{1, 12}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 13}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
net.layers{1, layer}.weights{1, 14}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
        
        end
end
opts.bias=0;
% 
% for ep=start_ep:opts.n_epoch
%     opts.ep=ep;
%     [net,opts]=train_net(net,opts);  
%     parameters=opts.parameters;
%     results=opts.results;
%     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
%     
%     opts.parameters.current_ep=opts.parameters.current_ep+1; 
% 
% % opts.train=[];
% % opts.test=[];
% end
% % % for ep=start_ep:opts.n_epoch
% % %     opts.ep=ep;
% % %     [net,opts]=train_net(net,opts); 
% % %     parameters=opts.parameters;
% % %     results=opts.results;
% % %     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
% % %     
% % %     opts.parameters.current_ep=opts.parameters.current_ep+1; 
% % % end
% % % for layer=1:numel(net.layers)
% % %       if isfield(net.layers{1,layer},'weights')
% % %           net.layers{1,layer}.weights{1,10}=sign(net.layers{1,layer}.weights{1,1});  
% % %            for number=1:numel(net.layers{1, layer}.weights)
% % %           net.layers{1, layer}.weights{1, number}=gather(net.layers{1, layer}.weights{1,number});
% % %            end
% % % %     　     net.layers{1, layer}.weights{1, 2}=gather(net.layers{1, layer}.weights{1, 2});
% % %             if any(layer==1)
% % % %             Weight=net.layers{1, layer}.weights{1, 1};
% % % %             net.layers{1, layer}.weights{1, 10}=Weight;
% % % %             net.layers{1, layer}.weights{1, 10}=sign(Weight);       
% % % %             net.layers{1, layer}.weights{1, 2}=zeros(size(net.layers{1, layer}.weights{1, 2}),'single');
% % %             
% % % %             pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1})*1;
% % % %             else
% % %             bits_size=fi(net.layers{1, layer}.weights{1, 1},1,8);
% % %             net.layers{1, layer}.weights{1, 1}=bits_size.single;
% % % %             pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1})*32;
% % %             end
% % %           net.layers{1, layer}.weights{1, number}=gpuArray(net.layers{1, layer}.weights{1,number});
% % %           net.layers{1, layer}.weights{1, 3}=logical(net.layers{1, layer}.weights{1, 1});
% % %       end
% % % end
% opts.pruning_sgd=sum(pruning_sgd,2);
%     opts.parameters.learning_method='sgd_or'; 
% start_ep=1;
% for ep=start_ep:opts.n_epoch
%     opts.ep=ep;
%     [net,opts]=train_net_bit(net,opts); 
%     parameters=opts.parameters;
%     results=opts.results;
%     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
%     
%     opts.parameters.current_ep=opts.parameters.current_ep+1; 
% end
%  rng(1);
opts.batchsize=randi(opts.n_train,1,opts.n_train); 
%  opts.parameters.l1_lr_5=1-0.6/(opts.n_train^(-2/7)+1);    %svrg
% opts.parameters.l1_lr_5=1-1/((3*opts.n_train^(1/5))/2);  %saga
iteration=0;
a=0.1;
opts.count=opts.n_train_batch;
opts.count1=0;
opts.epslon=10^(-19.1); %(-2) norm (10) cifar
opts.rp=10^(-18.1);
% if start_ep==1
%     opts.Dzdw=0;
% end
% opts.c=opts.n_train;
for ep=start_ep:opts.n_epoch
    opts.ep=ep;  
%     if ep>1
%     opts.order_batch_b=opts.n_train;
%     else 
%       if opts.ep<=1
%       opts.order_batch= opts.n_train;
%       end
%     opts.order_batch_c=double(round(((1*opts.Dzdw)/opts.epslon)^(32/27)));
%      if opts.order_batch_c==0
%           opts.order_batch_c=(opts.n_train);          
%      end
%      if  opts.order_batch_c>opts.n_train
%           opts.order_batch_c=opts.n_train-opts.n_test;
%      elseif opts.order_batch_c<=opts.n_test
% %          opts.order_batch_b=round(opts.n_train^(0.9));
%      opts.order_batch_c=opts.order_batch;
%      end

         opts.order_batch_b=double(round(((4.8*opts.Dzdw)/opts.epslon)^(32/27)));
%          if opts.order_batch_b>opts.c
%              opts.order_batch_b=opts.c;
%          end
% % %        opts.order_batch_b=double(round(((12*opts.Dzdw)/opts.epslon)));
% % %        opts.order_batch_b=opts.n_train;
% % %  opts.order_batch_b=double(round(((opts.ep^(3/2)*1000))));
       
    if opts.ep<=3
          opts.order_batch_b=(opts.n_train);          
    else
     if  opts.order_batch_b>=round(opts.n_train^(0.995))
          opts.order_batch_b=round(opts.n_train^(0.995));
     elseif opts.order_batch_b<=round(opts.n_train^(0.88))      
%          opts.order_batch_b=round(opts.n_train^(0.9));
     opts.order_batch_b=round(opts.n_train^(0.88));
     end
    end
% %     opts.order_batch_a=opts.n_train;
    opts.order_batch_a=double(round((opts.n_train^(2)*opts.Dzdw)/((opts.n_train*opts.Dzdw)+(opts.rp*(opts.n_train-1)^(3/2)*(0.325)^(2)))));
%     if opts.order_batch_a>opts.c
%              opts.order_batch_a=opts.c;
%     end
    if opts.ep<=3
          opts.order_batch_a=(opts.n_train);
     else
     if  opts.order_batch_a>=round(opts.n_train^(0.995))
          opts.order_batch_a=round(opts.n_train^(0.995));
     elseif opts.order_batch_a<=round(opts.n_train^(0.88))
         opts.order_batch_a=round(opts.n_train^(0.88));
     end
     end
     
    c=[opts.order_batch_b opts.order_batch_a];
      opts.order_batch=min(c);
%       opts.c=max(c);
%       if opts.c==opts.n_train
%           opts.c=min(c);
%       end
    if opts.order_batch==opts.n_train
         opts.order_batch=opts.n_train;
   opts.parameters.batch_size=round(opts.n_train^(0.625));
   opts.parameters.l1_lr_2=(1)/(3*0.02*(opts.n_train^(3/32)));
   opts.parameters.l1_lr_3=0;
      elseif opts.order_batch<opts.n_train
        if opts.order_batch_a==opts.order_batch_b
        opts.order_batch=opts.order_batch_b; 
         opts.parameters.batch_size=round(opts.order_batch^(0.625)); 
          opts.parameters.l1_lr_3=0.5;
          opts.parameters.l1_lr_2=(1)/(3*opts.sgd_lr_4*((opts.n_train)^(5/32)));
%% ours
%            if opts.order_batch>round(opts.n_train^0.85)
%   
%    opts.parameters.l1_lr_2=(1)/(3*opts.sgd_lr_4*((opts.n_train)^(3/32)));
%    opts.parameters.l1_lr_3=0.125;
%            else
%     
%         opts.parameters.l1_lr_2=(1)/(3*0.065*(opts.n_train^(3/32)));
%       opts.parameters.l1_lr_3=0.625;
%            end
         
         elseif opts.order_batch==opts.order_batch_b    
%% SCSG
% % %    opts.parameters.batch_size=round(opts.order_batch^(0.625));
% % %    opts.parameters.l1_lr_2=(1)/(6*opts.sgd_lr_4*opts.n_train^(1/4));
% % %    opts.parameters.l1_lr_3=0.5;
%% test 
% if opts.ep<=160
%     if opts.ep<=3
%      opts.order_batch=opts.order_batch_b   ;
%      opts.parameters.batch_size=round(opts.order_batch^(0.625));
%      opts.parameters.l1_lr_2=(1)/(3*opts.sgd_lr_4*opts.n_train^(3/32));
%          opts.parameters.l1_lr_3=0.0001;
%     else
% %      opts.order_batch=opts.order_batch_a;
%     end
% else
%  opts.order_batch=opts.order_batch_a;
% end
         opts.parameters.batch_size=floor(opts.order_batch^(0.625));
     opts.parameters.l1_lr_2=(1)/(3*opts.sgd_lr_4*((opts.n_train)^(3/32)));
     opts.parameters.l1_lr_3=0.75;
   elseif opts.order_batch==opts.order_batch_a
     opts.parameters.batch_size=floor(opts.order_batch^(0.625));
    opts.parameters.l1_lr_2=(1)/(3*opts.sgd_lr_4*((opts.n_train)^(5/32)));
     opts.parameters.l1_lr_3=0.4;
     
% %     elseif opts.order_batch_b==opts.order_batch_c
% %     opts.parameters.batch_size=round(opts.order_batch^(0.675));
% %     opts.parameters.l1_lr_2=(2)/(3*opts.sgd_lr_4*opts.order_batch^(3/32));
% %     opts.parameters.l1_lr_3=0.625;
     end
    end
%   end

%      if opts.order_batch<=opts.n_train^0.85
%        opts.parameters.batch_size=round(opts.order_batch^(0.625));
%         opts.parameters.l1_lr_2=(1)/(3*0.068*(opts.n_train^(3/32)));
%         if 
%         opts.parameters.l1_lr_3=0.8;
%      end

%       opts.parameters.l1_lr_2=opts.sgd_lr_2/(1.9*(opts.ep^(1.25)*opts.n_train_batch)^(1/2));
%      opts.results.order_batch_c(1,ep)=opts.order_batch_c;


% opts.parameters.l1_lr_2=(1)/(6*opts.sgd_lr_4*opts.n_train^(2/3));
%  opts.parameters.batch_size=round(opts.order_batch/32);
% opts.parameters.l1_lr_3=0.5;

     opts.results.order_batch_a(1,ep)=opts.order_batch_a;
     opts.results.order_batch_b(1,ep)=opts.order_batch_b;
     opts.results.mini_batch(1,ep)=opts.parameters.batch_size;
     opts.count1=opts.count1+1;
      opts.step=3;
        opts.xrange=1:opts.step:opts.n_epoch;
          
%     opts.l1_lr_1_b=opts.sgd_lr_2/(1.9*(opts.count)^(1/2));
% % % % opts.l1_lr_1_b=opts.sgd_lr_2/(1.9*(opts.ep^(1.25)*opts.n_train_batch)^(1/2));
% % % %    opts.l1_lr_1_a=(2)/(3*opts.sgd_lr_4*opts.n_train^(1/5));
% % % % %  opts.l1_lr_1_b=opts.sgd_lr_2/(1.9*(opts.n_epoch^(1.25)*opts.n_train_batch)^(1/2));
% % % %   opts.parameters.l1_lr_2=max(opts.l1_lr_1_a,opts.l1_lr_1_b);
  
 %
%  opts.parameters.l1_lr_2=(1)/(6*opts.sgd_lr_4*opts.n_train^(3/32));
   opts.Lamdba=1/(opts.n_epoch);
%    opts.parameters.l1_lr_2=2*opts.l1_lr_1_a;
 
% % % % % % %       if opts.parameters.l1_lr_2==opts.l1_lr_1_a
% % % % % % % %           opts.l1_lr_1_b=0.5; 
% % % % % % % %           opts.l1_lr_1_a=(1.5)/(a*3*opts.n_train^(1/5));
% % % % % % % %           opts.l1_lr_1_a1=opts.l1_lr_1_a;
% % % % % % % %           a=a+0.05;
% % % % % % % %           if opts.parameters.l1_lr_3<=0.5
% % % % % % % %          opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(0.55*opts.Lamdba);  
% % % % % % % %             if  opts.parameters.l1_lr_3<=0.5
% % % % % % % %              opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(1*opts.Lamdba); 
% % % % % % % %             if  opts.parameters.l1_lr_3<=0
% % % % % % %         if opts.ep==opts.count1+1
% % % % % % %          opts.parameters.l1_lr_3=0;
% % % % % % %        else
% % % % % % %          
% % % % % % % % % %           if opts.parameters.l1_lr_3<0.5
% % % % % % %          opts.parameters.l1_lr_3=0.3;
% % % % % % % % % %               if opts.parameters.l1_lr_3>0.5
% % % % % % % % % %                   opts.parameters.l1_lr_3=0.5;
% % % % % % % % % %               end
% % % % % % % % % %           else
% % % % % % % % % %               opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(0.5*opts.Lamdba);
% % % % % % % %                if opts.parameters.l1_lr_3<0.2
% % % % % % % %              opts.parameters.l1_lr_3=0.2;
% % % % % % % %               end
% % % % % % % % % %           end
% % % % % % %          
% % % % % % % %            end
% % % % % % % % 1/(opts.n_train^(1/3));
% % % % % % % %             end
% % % % % % %          end
% % % % % % % %            if  opts.parameters.l1_lr_3>=0.5
% % % % % % % %              opts.parameters.l1_lr_3=0.5;            
% % % % % % % %            end
% % % % % % %               opts.step=1;
% % % % % % %       elseif opts.parameters.l1_lr_2==opts.l1_lr_1_b
% % % % % % % %              if opts.ep<=1
% % % % % % %             
% % % % % % %           opts.parameters.l1_lr_3=opts.parameters.l1_lr_6;
% % % % % % % %              else
% % % % % % % %                opts.parameters.l1_lr_3=opts.parameters.l1_lr_6; 
% % % % % % % % %              if opts.ep==7
% % % % % % % % %                 opts.parameters.l1_lr_3=0.2;
% % % % % % % %              else
% % % % % % % %           opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(0.5*opts.Lamdba);
% % % % % % %          
% % % % % % % % % % %                
% % % % % % % %             if  opts.parameters.l1_lr_3<=0.2
% % % % % % % %              opts.parameters.l1_lr_3=0.2;   
% % % % % % % %             end
% % % % % % % %           opts.lambda=opts.parameters.l1_lr_3;    
% % % % % % % %              end
% % % % % % % %              end
% % % % % % %             opts.count1=opts.count1+1;
% % % % % % %         opts.step=3; 
% % % % % % %       end     
%             else
%             if opts.ep>=10    
% %           opts.parameters.l1_lr_3=0.67;
% %             else
%         opts.parameters.l1_lr_3=opts.parameters.l1_lr_3+(3*opts.Lamdba);
%           if  opts.parameters.l1_lr_3>=0.7
%              opts.parameters.l1_lr_3=0.7;        
            

%        end
%         end
           iteration=iteration+1;
%end 
     
   
      

%    end
% opts.parameters.l1_lr_2=2*opts.parameters.l1_lr_2;
%   opts.parameters.l1_lr_1=opts.l1_lr_1_a;
%    for t=0: m-1
%     opts.t=t;

% if  opts.parameters.l1_lr_2==opts.l1_lr_1_b 
%     if opts.parameters.l1_lr_3>=0.69 && opts.parameters.l1_lr_3<=0.8

%     if opts.ep==1
%         opts.sgd_lr_3=0;
%     end
%     opts.parameters.l1_lr_3=opts.sgd_lr_3+(opts.ep/opts.n_train_batch)^(3.5);
  
%   if opts.parameters.l1_lr_2==opts.l1_lr_1_b
%      opts.order_batch=round((2*opts.n_train^(2))/(2*opts.n_train+(1-0.8)*opts.n_train^(3/2)*0.1));

%     opts.order_batch=opts.n_train;
 %   opts.order=randperm(opts.order_batch);
 %   opts.order=opts.batchsize(1:opts.order_batch);
%     else
%     opts.order_batch=round((2*opts.n_train^(2))/(2*opts.n_train+(0.1)*opts.n_train^(3/2)*0.1)); 
 
 %   opts.order=randperm(opts.order_batch);
%    opts.order=opts.batchsize(1:opts.order_batch);
%   end
 
%     if  mod(opts.order_batch,opts.parameters.batch_size)~=0
%     a_mod=mod(opts.order_batch,opts.parameters.batch_size);
%     opts.order_batch=opts.order_batch-a_mod;
%     end
%      opts.order=opts.batchsize((1+(ep-1)*opts.order_batch:ep*opts.order_batch));
%  rng(2)

    opts.order=randperm(opts.n_train,opts.order_batch);
%     opts.order=randperm(opts.n_train); 
      
    opts.leftorder=setxor(opts.order,opts.batchsize);
%     if  mod(opts.leftorder,opts.parameters.batch_size)~=0
%     b_mod=mod(length(opts.leftorder),opts.parameters.batch_size);
%     opts.leftorder=opts.leftorder(1,1:(length(opts.leftorder)-b_mod));
%     end
    if length(size(opts.train))>=3
    opts.train_batch=opts.train(:,:,:,opts.order(:));
    opts.train_batch_left=opts.train(:,:,:, opts.leftorder);
    else
     opts.train_batch=opts.train(:,opts.order(:));  
     opts.train_batch_left=opts.train(:,opts.leftorder);
    end 
    signal=ismember(opts.ep,opts.xrange);       
%   if signal==1
%       opts.n_train_batch=opts.n_train/opts.parameters.batch_size;
%       if isfield(opts,'n_valid')
%     opts.n_valid_batch=round(opts.n_train/opts.parameters.batch_size);
%       end
% 
%   else
    opts.n_train_batch=opts.order_batch/opts.parameters.batch_size;
    if isfield(opts,'n_valid')
    opts.n_valid_batch=(opts.order_batch/opts.parameters.batch_size);
    end

%   end 
    
     %end
%end
 opts.q=round(opts.n_train^(0.2));
 [net,opts]=train_net(net,opts);

%      else
%          
% [net,opts]=train_net_saga(net,opts);
%     end
 
%     end 
%           for layer=1:numel(net.layers)
%          if isfield(net.layers{layer},'weights')&&~isempty(net.layers{layer}.weights)  
%                     net.layers{1, layer}.weights{1, 6}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%          end
%           end
%  elseif  opts.parameters.l1_lr_2==opts.l1_lr_1_a
% %     if opts.parameters.l1_lr_3>=0.67 && opts.parameters.l1_lr_3<=0.75
%      opts.step=1;
%      opts.xrange=1:opts.step:opts.n_epoch;
%       [net,opts]=train_net(net,opts);  
% %     else
% %      [net,opts]=train_net_saga(net,opts); 
% %     end
%  end
%       opts.xrange=ep:5:opts.n_epoch;
% %     [net,opts]=train_net(net,opts);   
%     end
% end
%   for layer=1:numel(net.layers)
%     if isfield(net.layers{1,layer},'weights')   
%     net.layers{1, layer}.weights{1, 13}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%     end
%   end
  

    parameters=opts.parameters;
    for layer=1:numel(net.layers)
       if isfield(net.layers{1,layer},'weights')
%          net.layers{1, layer}.weights{1, 6}=zeros(size(net.layers{1, layer}.weights{1, 1}),'single');
%      if t==m-1
%       net.layers{1, layer}.weights{1,7}=net.layers{1, layer}.weights{1,11};
%       net.layers{1, layer}.weights{1,9}=net.layers{1, layer}.weights{1,10};
%      end
%            net.layers{1, layer}.weights{1,6}=0;
          if any(layer==1) 
             pruning_sgd(1,layer)= nnz(net.layers{1, layer}.weights{1, 1});
          else
             pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1});
          end
       end
   end
    opts.pruning_sgd=sum(pruning_sgd,2);  
    opts.parameters.current_ep=opts.parameters.current_ep+1; 
%  end

% opts.train=[];
% opts.test=[];

if isfield(opts, 'valid')&&(numel(opts.valid)>0)
    opts.validating=1;
[opts]=test_net(net,opts);    
end

opts.validating=0;
% if any(layer==1) || any(layer==12) || any(layer==4)
%      [opts]=test_net(net,opts);
%  else
     [opts]=test_net(net,opts);
%  end
%  for layer=1:numel(net.layers)
%       if isfield(net.layers{1,layer},'weights')
%             if any(layer==4) || any(layer==7)
%    pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1})*1;
%                else 
%    pruning_sgd(1,layer)=nnz(net.layers{1, layer}.weights{1, 1})*(opts.fcc_bits-1);
%             end
%       end
%  end 
 %        end
     if opts.plot
        %figure(figure1);
        if strcmp(net.layers{end}.type,'softmaxloss')
            subplot(1,5,1); 
            plot(opts.results.TrainEpochError,'b','DisplayName','Train (top1)');hold on;
            plot(opts.results.TrainEpochError_Top5,'b--','DisplayName','Train (top5)');hold on;
            if isfield(opts,'valid')&&(numel(opts.valid)>0)
                plot(opts.results.ValidEpochError,'g','DisplayName','Valid (top1)');hold on
                plot(opts.results.ValidEpochError_Top5,'g--','DisplayName','Valid (top5)');hold on;
            end
            plot(opts.results.TestEpochError,'r','DisplayName','Test (top1)');hold on;
            plot(opts.results.TestEpochError_Top5,'r--','DisplayName','Test (top5)');hold off;
            
            title('Error Rate per Epoch');legend('show');
            subplot(1,5,2); 
            plot(opts.results.TrainEpochLoss,'b','DisplayName','Train');hold on;
            if isfield(opts,'valid')&&(numel(opts.valid)>0)
                plot(opts.results.ValidEpochLoss,'g','DisplayName','Valid');  hold on;          
            end            
            plot(opts.results.TestEpochLoss,'r','DisplayName','Test');hold off;
            title('Loss per Epoch');legend('show');
            
            subplot(1,5,3)
            plot(opts.tElapsed);
            title('time');legend('show')
            
%             if opts.ep>=10
            subplot(1,5,4)
%             plot(opts.results.order_batch_c,'DisplayName','Batchsize'); hold on;
            plot(opts.results.order_batch_b,'b','DisplayName','Batchsize-b'); hold on;
            plot(opts.results.order_batch_a,'r','DisplayName','Batchsize-a'); hold off;
            title('Batchsize');legend('show')                  
%             end
            
            subplot(1,5,5)
            plot( opts.results.mini_batch,'DisplayName','MiniBatchsize');
            title('Loss per Epoch');legend('show')   
            drawnow;
         end
        
    end
    
    parameters=opts.parameters;
    results=opts.results;
%     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
    
%      end
end
opts.train=[];
opts.test=[];
 if strcmp(net.layers{end}.type,'softmaxloss')
    if isfield(opts,'valid')&&(numel(opts.valid)>0)
        [min_err_valid,best_id]=min(opts.results.ValidEpochError);
         min_err=opts.results.TestEpochError(best_id);  
         disp(['Model validation error rate: ',num2str(min_err_valid)]);
    else
        [min_err,best_id]=min(opts.results.TestEpochError);
    end
    disp(['Model test error rate: ',num2str(min_err)]);
    best_net_source=[fullfile(opts.output_dir,[opts.output_name,num2str(best_id),'.mat'])];
    best_net_destination=[fullfile(opts.output_dir,['best_',opts.output_name,num2str(best_id),'.mat'])];
%     copyfile(best_net_source,best_net_destination);
 end
% saveas(gcf,[fullfile(opts.output_dir,[opts.output_name,num2str(opts.n_epoch),'.pdf'])])


