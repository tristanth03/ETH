function r=wintp1(x,y,z,w)
% interpolate with a window of size w on z.
if nargin<4, w=length(z); end;
m=length(z); n=floor(m/w); r=[];
for i=1:n
 dz=z((i-1)*w+1:i*w); k=interval(x,dz); 
 t=intpol(x(k),y(k),dz); r=[r;t];
end;
if (rem(m,w)~=0),
	e=rem(m,w); g=[n*w+1:n*w+e]; dz=z(g);k=interval(x,dz);
	t=intpol(x(k),y(k),dz);	r=[r;t];
end;

function i=interval(x,y)
% find the x: min(x)<min(y) and max(x)> max(y)
i1=find(x<min(y)); i2=find(x>max(y)); i=i1(size(i1,1)):i2(1);
j=i(1); while (x(j)>y(1)), j=j-1; end;
k=max(i); while(x(k)<max(y)),	k=k+1; end;
i=[(j:i(1)-1)';i;(max(i)+1:k)'];

function r=intpol(x,y,z);
% linear interp of z to f(x,y), needs min(x)< z < max(x)
n=length(z); r=[]; j=1; i=1;
while i<=n, flag=i;
  while (flag==i)
    if (z(i)>=x(j) & z(i)<x(j+1))
      r=[r;y(j+1)-(x(j+1)-z(i))*(y(j+1)-y(j))/(x(j+1)-x(j))];     
	    i=i+1;
    else, j=j+1; end;
  end;
end;
