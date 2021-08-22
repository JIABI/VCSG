

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
    opts.n_valid_batch=floor(opts.n_valid/opts.parameters.batch_size);
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
 opts.parameters.l1_lr_5=1-0.6/(opts.n_train^(-2/7)+1);    %svrg
% opts.parameters.l1_lr_5=1-1/((3*opts.n_train^(1/5))/2);  %saga
iteration=0;
a=0.3;
for ep=start_ep:opts.n_epoch
   opts.ep=ep;    
   opts.l1_lr_1_a=(0.8)/(0.35*3*opts.n_train^(1/5));
  opts.l1_lr_1_b=opts.sgd_lr_2/((opts.ep^(1.85)*opts.n_train_batch)^(1/2));
  opts.parameters.l1_lr_2=max(opts.l1_lr_1_a,opts.l1_lr_1_b);
 % opts.l1_lr_1_b=opts.sgd_lr_2/(1*(opts.n_epoch*opts.n_train_batch)^(1/2));
% opts.parameters.l1_lr_2=opts.l1_lr_1_b;
   opts.Lamdba=1/(opts.n_epoch);
%    opts.parameters.l1_lr_2=2*opts.l1_lr_1_a;
 
      if opts.parameters.l1_lr_2==opts.l1_lr_1_a
%           if opts.parameters.l1_lr_3<=0.5
%           opts.parameters.l1_lr_3=opts.parameters.l1_lr_3-(0.25*opts.Lamdba);  
%             if  opts.parameters.l1_lr_3<=0.3
             opts.parameters.l1_lr_3=0.5; % 1/(opts.n_train^(1/3));
%             end
%           end
%            if  opts.parameters.l1_lr_3>=0.5
%              opts.parameters.l1_lr_3=0.5;            
%            end
      elseif opts.parameters.l1_lr_2==opts.l1_lr_1_b
           
         opts.parameters.l1_lr_3=0.5;
 
       end          
   
      
%     end
% opts.parameters.l1_lr_2=2*opts.parameters.l1_lr_2;
%   opts.parameters.l1_lr_1=opts.l1_lr_1_a;
%    for t=0: m-1
%     opts.t=t;

% if  opts.parameters.l1_lr_2==opts.l1_lr_1_b 
%     if opts.parameters.l1_lr_3>=0.69 && opts.parameters.l1_lr_3<=0.8
 opts.step=5;
 opts.xrange=1:opts.step:opts.n_epoch;
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
%     end
    opts.pruning_sgd=sum(pruning_sgd,2);  
    opts.parameters.current_ep=opts.parameters.current_ep+1; 
end

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
%         end
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
            plot(opts.fx(1,:));
            title('Loss per Epoch');legend('show')
            
%             if opts.ep>=10
%             subplot(1,5,4)
%             plot(opts.fx1(1,:));
%             title('Loss per Epoch');legend('show')                  
%             end
            
%             subplot(1,5,5)
%             plot(opts.weight(1,:));
%             title('Loss per Epoch');legend('show')   
            drawnow;
        end
        
    end
    
    parameters=opts.parameters;
    results=opts.results;
%     save([fullfile(opts.output_dir,[opts.output_name,num2str(ep),'.mat'])],'net','parameters','results');     
    
    opts.parameters.current_ep=opts.parameters.current_ep+1;
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


