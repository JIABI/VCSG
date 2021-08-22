 function imdb = getImageNetImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile('/home/bonnie/LightNet-master', 'ILSVRC');
files=[{'train.mat'},{'test.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 1), 3]);

% if any(cellfun(@(fn) ~exist(fn, 'file'), files))
%   url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz' ;
%   fprintf('downloading %s\n', url) ;
%   untar(url, opts.dataDir) ;
% end

data = cell(1, numel(files));
labels = cell(1, 2);
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
% %   if fi==1
% %   data{fi} = permute(reshape(fd.train_data,96,96,3,[]),[2 1 3 4]) ;
% %   elseif fi==2
% %   data{fi} = permute(reshape(fd.test_data,96,96,3,[]),[2 1 3 4]) ;  
% %   labels{2} = fd.test_label'; % Index from 1
% %   elseif fi==3
% %   labels{1} = fd.train_label'; % Index from 1
% %   end
% %  sets{fi} = repmat(file_set(fi), size(labels{1}));
if fi==1
    fd.data=fd.train_data;
    fd.labels=fd.train_label;
elseif fi==2
    fd.data=fd.test_data;
    fd.labels=fd.test_label;
end
  data{fi} = permute(fd.data,[2 1 3 4]) ;
%  data{fi}= imresize(Data{fi},[96,96]);
  labels{fi} = fd.labels; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));

end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

  if isfield(opts,'contrastNormalization')&&opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n,30)) ;
  data = reshape(z, 96, 96, 3, []) ;
  end

if isfield(opts,'whitenData') &&opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

% clNames = load(fullfile(unpackPath, 'meta.mat'));
imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
% imdb.meta.sets = {'train','test'} ;
% imdb.meta.classes = clNames.fine_label_names;