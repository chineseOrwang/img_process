function [data]=chafen(J)
A=double(J);
[m,n]=size(A);                         %��A����Ĵ�С����ֵ��m n
B=A;
C=zeros(m,n);                         %�½�ȫ�����C����������һ���ĻҶ�ֱ��ͼ
for i=1:m-1
    for j=1:n-1
        B(i,j)=A(i+1,j+1);
        C(i,j)=(round(A(i,j)-B(i,j)));%Դ����ΪC(i,j)=abs(round(A(i,j)-B(i,j))); �����Ҿ����ɹ�ʽ��1���ļ���Ӧ�ò���abs.
    end
end
h=imhist(mat2gray(C))/(m*n); %��һ���ĻҶ�ֱ��ͼ
mean=0;con=0;ent=0;                    % ��ֵmean���Աȶ�con����ent��ʼֵ����
for i=1:256                              %ѭ������ֵmean���Աȶ�con����ent
    mean=mean+(i*h(i))/256;
    con=con+i*i*h(i);
    if(h(i)>0)
        ent=ent-h(i)*log2(h(i));
    end
end
a= mean;
b=con;
c=ent;
data=[a',b',c'];
end