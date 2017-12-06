function beta = quantreg2 (q, p, y, x, toler, beta, flb, fub, fhandle, dfhandle)
% This function was slightly modified by Kyle Perline on 11/2/16
% it only does linear quantile regression (see 'func' argument that has
% been removed)
%
% function beta = quantreg2 (q, p, y, x, func, toler, beta)
%   Use interior point algorithm for quantile regression.
%   (See Koenker and Park (1996), J. Econometrics) 
%
%  Inputs:
%     -- q is the quantile desired (number in (0,1)).
%     -- p is the size of the parameter vector.  This is actually
%        redundant since it's just the length of beta, which is also
%        input, but I didn't change the code to eliminate this 
%        redundancy.  Note that p is not necessarily the number of columns
%        of x, though this is usually the case.
%     -- y is the response vector (n x 1).
%     -- x is the predictor matrix (n x m).
%     -- func is a character array containing the name of a function
%        that evaluates f(x, beta).  This MATLAB function takes (x, beta)
%        as inputs where x is the n x m predictor matrix and beta is some
%        p x 1 vector of parameter values.  It returns an n x 1 vector.
%        In the case of linear QR, m=p and the function (whatever it's
%        called) returns x*beta.  
%        There must also be a second MATLAB function, with the same name
%        as func except with the letter d appended to the front, that
%        also takes (x,beta) as input but returns the $n x p$ matrix
%        of partial derivatives of f(x,beta) evaluated at beta. In the
%        linear QR example, m=p and this second function simply returns x.
%     -- toler is a small number giving the minimum change in the value
%        of the surrogate function before convergence is declared.
%     -- beta is the starting value of beta (p x 1 vector).
%
%        Kyle Perline update
%     -- flb is the (n X 1) lower bound so that the quantile =
%        max(flb,func(x,beta))
%     -- fub is the (n X 1) upper bound so that the quantile =
%        min(fup,func(x,beta))

% flops(0);                      % reset flops count.
% dfunc = ['quantreg_d' func];            % name of differential function.
% func  = ['quantreg_' func];
eta=.97;   % Value recommended by K&P
omq=1-q;
% flops2=0;

iteration = 0;
n=length(y);
d=zeros(n,1);
change = realmax;

if isempty(fhandle)
    fhandle = @(x,beta) x*beta;
    dfhandle = @(x,beta) x;
end

% residuals = y-feval(func,x,beta);
flb = -999*ones(size(flb));
fub =  999*ones(size(fub));
residuals = y-min(max(fhandle(x,beta),flb),fub); % Kyle Perline update - linear
lastobj=q*sum(residuals)-sum(residuals(residuals<0));

while abs(change) > toler 
	iteration = iteration + 1;
% 	J=feval(dfunc,x,beta);   % J is first differential matrix.
    % Kyle Perline update
    % the partial derivatives are x if flb<=x*beta<=fub and 0 otherwise
    J = dfhandle(x,beta); % Kyle Perline update
    f = fhandle(x,beta);
    J( flb>f | f>fub , :) = 0;
	dhat=d-J*(inv(J'*J)*(J'*d));
	m=toler+max([-dhat./(1-q);dhat./q]);
	d = dhat ./ m;

% Now we perform the inner iterations.  K&P recommend two of them per
% outer iteration.
	for k=1:2  
		dmatrix = min([q-d  1-q+d]')';
		d2=dmatrix.^2;
		direc = inv(J'*(d2(:,ones(p,1)).*J))*(J'*(d2.*residuals));
		s = d2.*(residuals - J*direc);
		alpha = max(max([s./(q-d) -s./(1-q+d)]'));
		d=d+eta/alpha*s;
	end

% Now we do the line search recommended by K&P.  Some of the parameters
% passed to the fmin function are ad hoc (K&P give no specific 
% recommendations).
%     callfunc = @(t)quantreg2_helper(t,func,q,y,x,beta',direc);
    callfunc = @(t)quantreg2_helper(t,q,y,x,beta,direc,fhandle); % Kyle Perline update
    step = fminbnd(callfunc,-1,1) * direc;
% 	step=fminsearch('ipqr_objfunc',-1,1,[],func,q,y,x,beta,direc) * direc;

	beta = beta + step;
% 	residuals = y - feval(func, x, beta);
    residuals = y - min(max(fhandle(x,beta),flb),fub); % Kyle Perine update
	obj=q*sum(residuals)-sum(residuals(residuals<0));
	change=lastobj-obj;
	lastobj=obj;
end

if nargout==0
    
    subplot(2,1,1)
    scatter(x,y,'b.')
    hold on
    x2 = linspace(min(x),max(x),500);
    y2 = zeros(size(x2));
    for ii=1:length(y2)
%         y2(ii) = feval(func,x2(ii),beta);
        y2(ii) = fhandle(x2(ii),beta); % Kyle Perline update
    end
    plot(x2,y2,'r.')
    legend('data',sprintf('quantreg-fit ptile=%.0f%%',q*100),'location','best')
    hold off
    subplot(2,1,2)
    tmp = sortrows([x,y]);
    x2 = tmp(:,1);
    y2 = tmp(:,2);
    ybeta = zeros(size(y2,1),size(y2,2));
    for ii=1:length(y2)
%         ybeta(ii) = feval(func,x2(ii),beta);
        ybeta(ii) = fhandle(x2(ii),beta); % Kyle Perline update
    end
    greater = ybeta>y2;
    navg = 15;
    y3 = zeros(length(y2)-2*navg,1);
    for ii=1:length(y3)
        y3(ii) = mean(greater(ii:ii+2*navg));
    end
    subplot(2,1,2)
    plot(x2(navg+1:end-navg),y3)
%     y2 = zeros(size(x));
%     for ii=1:length(y2)
%         y2(ii) = feval(func,x(ii),beta);
%     end
%     tmp = sortrows([x,y,y2],1);
%     greater = tmp(:,3)>=tmp(:,2);
%     navg = 5;
%     x2 = tmp(navg+1,end-navg-1);
%     y2 = zeros(length(x2),1);
%     for ii=1:length(x2)
%         y2(ii) = mean(greater(ii-navg:ii+navg));
%     end
%     plot(x2,y2)
%     return
end 

%  At this point, beta is the estimate of regression coefficients;
%                 lastobj is the value of the objective function at beta;
%                 flops is the number of floating-point operations needed
%                        (according to the way MATLAB counts them, at least).