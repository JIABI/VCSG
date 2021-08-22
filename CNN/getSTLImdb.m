function imdb = getSTLImdb(opts)
% -------------------------------------------------------------------------
%Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'stl10_matlab');
% files = [ {'train.mat'}, {'test.mat'}, {'unlabeled.mat'}];
files = {'train', ...
         'test'} ;

files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1,1),3]);

%if any(cellfun(@(fn) ~exist(fn, 'file'), files))
%  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
%  fprintf('downloading %s\n', url) ;
%  untar(url, opts.dataDir) ;
%end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.X',96,96,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.y'; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end
set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1),4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if isfield(opts,'contrastNormalization')&&opts.contrastNormalization
  z = reshape(data,[],13000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 96, 96, 3, []) ;
end

if isfield(opts,'whitenData') &&opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/13000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 96, 96, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'stl_meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.class_names;