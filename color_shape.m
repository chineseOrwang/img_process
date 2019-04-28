function [c_s]=color_shape(a)
b=rgb2hsv(a);
h=b(:,:,2);
[m,n]=size(h);
for i=1:m
    for j=1:n
        if h(i,j)<0.1
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
    end
end
ra=0;
ga=0;
ba=0;
zk=0;
for i=1:m
    for j=1:n
        if a(i,j,1)>0&&a(i,j,2)>0&&a(i,j,3)>0
            zk=zk+1;
            ra(zk)=a(i,j,1);
            ga(zk)=a(i,j,2);
            ba(zk)=a(i,j,3);
        end
    end
end
color=[mean(ra),mean(ga),mean(ba),(sum((ra-mean(ra)).^2)/zk)^(1/2),(sum((ga-mean(ga)).^2)/zk)^(1/2),(sum((ba-mean(ba)).^2)/zk)^(1/2),(sum((ra-mean(ra)).^3)/zk)^(1/3),(sum((ga-mean(ga)).^3)/zk)^(1/3),(sum((ba-mean(ba)).^3)/zk)^(1/3)];
zl=0;
for i=2:(m-1)
    for j=1:n
        if h(i,j)>=0.1&&(h(i+1,j)<0.1||h(i-1,j)<0.1)
            zl=zl+1;
        end
    end
end
for i=1:m
    for j=1:n
        if a(i,j,1)>255
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
        if a(i,j,1)<150
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
        if a(i,j,2)>255
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
        if a(i,j,2)<90
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
        if a(i,j,3)>150
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
        if a(i,j,3)<20
            a(i,j,1)=0;
            a(i,j,2)=0;
            a(i,j,3)=0;
        end
    end
end
k=0;
for i=1:m
    for j=1:n
        if a(i,j,1)>0&&a(i,j,2)>0&&a(i,j,3)>0
            k=k+1;
        end
    end
end
l=0;
for i=2:(m-1)
    for j=1:n
        if a(i,j,1)>0&&a(i,j,2)>0&&a(i,j,3)>0
            if (a(i+1,j,1)==0&&a(i+1,j,2)==0&&a(i+1,j,3)==0)||(a(i-1,j,1)==0&&a(i-1,j,2)==0&&a(i-1,j,3)==0)
                l=l+1;
            end
        end
    end
end
shape=[(l-zl)/zk,k/zk,4*pi*k/(l-zl)^2];
c_s=[color,shape];
end