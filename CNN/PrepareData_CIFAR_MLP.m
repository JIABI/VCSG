function [ opts ] = PrepareData_CIFAR_MLP( opts )


imdb=getCifarImdb(opts);

opts.train=imdb.images.data(:,:,:,imdb.images.set==1);
opts.train=reshape(opts.train,prod(size(opts.train))/nnz(imdb.images.set==1),nnz(imdb.images.set==1));
opts.train_labels=imdb.images.labels(imdb.images.set==1);
opts.test=imdb.images.data(:,:,:,imdb.images.set==3);
opts.test=reshape(opts.test,prod(size(opts.test))/nnz(imdb.images.set==3),nnz(imdb.images.set==3));
opts.test_labels=imdb.images.labels(imdb.images.set==3);

opts.n_class=max(opts.train_labels(:));
opts.n_train=size(opts.train,2);
opts.n_test=size(opts.test,2);

end

