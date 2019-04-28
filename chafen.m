function [data]=chafen(J)
A=double(J);
[m,n]=size(A);                         %求A矩阵的大小，赋值给m n
B=A;
C=zeros(m,n);                         %新建全零矩阵C，以下求解归一化的灰度直方图
for i=1:m-1
    for j=1:n-1
        B(i,j)=A(i+1,j+1);
        C(i,j)=(round(A(i,j)-B(i,j)));%源程序为C(i,j)=abs(round(A(i,j)-B(i,j))); 但是我觉得由公式（1）的计算应该不加abs.
    end
end
h=imhist(mat2gray(C))/(m*n); %归一化的灰度直方图
mean=0;con=0;ent=0;                    % 均值mean、对比度con和熵ent初始值赋零
for i=1:256                              %循环求解均值mean、对比度con和熵ent
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