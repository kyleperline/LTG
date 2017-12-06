function value = quantreg2_helper (t,q,y,x,start,dir,fhandle)
% function value = quantreg2_helper (t,func,q,y,x,start,dir)
% This function was slightly modified by Kyle Perline on 11/2/16
% it only does linear quantile regression (see 'func' argument that has
% been removed)
%
% Calculate objective function at start+t*dir. 

% res=y-feval(func,x,start+t*dir);  % Find residuals for proposed beta.
res=y-fhandle(x,start+t*dir); % Kyle Perline update
value=q*sum(res) - sum(res(res<0));