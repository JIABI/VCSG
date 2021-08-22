function[y]=BatchNorm(I,L1,L2,m)
miu=mean(I);
gamma=(1/m)*((I-miu).^(2));
I1=(I-miu)./sqrt(gamma);
y=L1*I1+L2;

end