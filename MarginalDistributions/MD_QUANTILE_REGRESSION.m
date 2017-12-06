classdef MD_QUANTILE_REGRESSION < MARGINAL_DISTRIBUTIONS
    % This class performs quantile regression for estimating the marginal
    % distributions
    %
    % Main Idea:
    %  Given predictor variables X (nXd), response variables y (nX1), and
    %  quantile level 0<=tau<=1, calculate beta (dX1) such that with 
    %  yhat = X*beta, yhat is the tau-quantile estimator
    %  This looks linear at first, but additional columns (predictor
    %  variables) can be added to X to make it non-linear
    %
    %  A finite number of quantiles are estimated, so some sort of
    %  regression must be used to estimate all quantiles.
    %  Let the quantiles that are estimated be q1<q2<...<qn
    %  A new quantile q s.t. q1<q<qn is estimated with piecewise linear
    %  interpolation
    %  The tails are interpolated with an exponential tail, so if 0<q<q1
    %  then q has an exponential distribution between zerop and q1
    %
    properties(Constant = true)
        tau_list = 0.05:0.05:0.95; % list of quantiles that are estimated
                                   % 0<tau_list(1), tau_list(end)<1
    end
    properties
       beta % dXlength(quant_list) regression values
            % X*beta(:,ii) is the estimator of quant_list(ii) quantile
       method % string, type of regression to perform
       QRB % QR_BASIS instance, optional. replaces method
       qrb_handle
       qrb_dhandle
    end
    
    methods
        function obj = MD_QUANTILE_REGRESSION(method)
            % input the method 
            obj.method = method;
            obj.qrb_handle = @(x,beta) x*beta;
            obj.qrb_dhandle = @(x,beta) x;
            obj.clear_train();
        end
        function d = get_description(obj)
            d = ['QR_',obj.method];
        end
        function name = get_name(obj)
            name = ['QR_',obj.method];
        end
        function obj = set_method(obj,method)
            obj.method = method;
        end
        function obj = input_qrb(obj,qrbinstance)
            obj.QRB = qrbinstance;
            [obj.qrb_handle, obj.qrb_dhandle] = ...
                qrbinstance.get_basis_function_handles();
        end
        function obj = train(obj, prev_MD_training)
            if nargin<2
                prev_MD_training = [];
            end
            obj.train_helper();
            % obj.X and obj.y may have NaNs. need to remove these
            nanrowsX = sum(isnan(obj.X),2)>0;
            nanrowsy = sum(isnan(obj.X),2)>0;
            notnanrows = ~(nanrowsX | nanrowsy);
            X = obj.X(notnanrows,:);
            y = obj.y(notnanrows);
            
            % first get the transformed X
            Xtransformed = obj.transformX(X);
            p = size(Xtransformed,2);
            % for each quantile tau in tau_list we need to estimate beta
            if isempty(prev_MD_training)
                if isempty(obj.beta)
                    if isempty(obj.QRB)
                        obj.beta = ones(p,length(obj.tau_list))/p;
                    else
                        obj.beta = obj.QRB.get_binit()*ones(1,length(obj.tau_list));
                        p = obj.QRB.get_p();
                    end
                end
            else
                obj.beta = prev_MD_training;
            end
            for ii=1:length(obj.tau_list)
                % if beta has already been estimated (i.e. train(obj) has
                % already been called), then use beta as the initial guess;
                % this helps calculate beta faster.
                % otherwise, just make a random initial point
                binit = obj.beta(:,ii);
                flb = obj.zerop(X);
                fub = obj.onep(X);
                obj.beta(:,ii) = quantreg2(obj.tau_list(ii),p,...
                    y,Xtransformed,1e-6,binit,flb,fub,...
                    obj.qrb_handle,obj.qrb_dhandle);
            end
        end
        function training = get_training(obj)
            training = obj.beta;
        end
        function obj = clear_train(obj)
            obj.beta = [];
        end
        function p = cdf(obj,X,y)
            % get quantile estimates
            % quant(ii,jj) is the taus(jj) quantile estimate of X(ii,:)
            % taus is strictly increasing, taus(1)=0, taus(end)=1
            obj.cdf_helper(X,y);
            % first get rid of rows with NaNs
            nanrows = any(isnan(X),2) | any(isnan(y),2);
            nrows = size(X,1);
            X = X(~nanrows,:);
            y = y(~nanrows,:);
            [quant,taus] = obj.get_quantiles(X);
            % do linear interpolation
            p2 = zeros(size(y,1),1);
            for ii=1:size(y,1)
                p2(ii) = interp1(quant(ii,:),taus,y(ii));
            end
            % now put NaNs back in place
            p = zeros(nrows,1);
            p(~nanrows) = p2;
            p(nanrows)  = NaN;
        end
        function y = inverse_cdf(obj,X,p)
            % get inverse quantile
            % quant(ii,jj) is the taus(jj) quantile estimate of X(ii,:)
            % taus is strictly increasing, taus(1)=0, taus(end)=1
            assert(size(X,1)==size(p,1) || all(size(p)==[1,1]),'X or p are wrong size')
            if size(p,1)==1
                p = ones(size(X,1),1)*p;
            end
            [quant,taus] = obj.get_quantiles(X);
            % do linear interpolation
            y = zeros(size(X,1),1);
            for ii=1:size(X,1)
                y(ii) = interp1(taus,quant(ii,:),p(ii));
            end
            obj.inverse_cdf_helper(X,y);
        end
        function [quant,taus] = get_quantiles(obj,X)
            % for each tau in tau_list estimate the quantiles of each data
            % point (row) in Xtransformed
            % inputs:
            %  X: mXd predictor array
            % returns:
            %  quant: mX(length(tau_list)+2*ntail) array of quantile
            %         estimates 
            %         ntail>0 is the number of tail quantiles to estimate
            %         quant(jj,ii+ntail) is the tau_list(ii) quantile 
            %         estimate of X(jj,:)
            %  taus:  the list of quantiles. taus(ii+ntail)=tau_list(ii),
            %         taus(1) = 0, taus(end) = 1
            
            ntail = 5;
            % first get transformed X
            Xtransformed = obj.transformX(X);
            % get quantile estimates
            quant = zeros(size(X,1),length(obj.tau_list));
            for ii=1:length(obj.tau_list)
                quant(:,ii) = obj.qrb_handle(Xtransformed,obj.beta(:,ii));
            end
            % quantile fudging - there are three requirements quant must
            % satisfy that we need to manually enforce. This requires a
            % small constant eps
            delta = obj.onep(X)-obj.zerop(X);
            epsvec = min(1e-5,delta/(length(obj.tau_list)+2));
            % 1. zerop(X) < quant(:,ii)
            %    the inequality is strict since we need the cdf to be
            %    strictly increasing
            %    since zerop(X) is fixed, we must move quant(:,ii)
            
            quant(:,1) = max(obj.zerop(X)+epsvec,quant(:,1));
            % 2. quant(:,end) < onep(X)
            %    same deal as step 1
            quant(:,end) = min(obj.onep(X)-epsvec,quant(:,end));
            % 3. quant(:,ii) < quant(:,ii+1)
            %    this is because we need strictly increasing cdf
            for ii=2:length(obj.tau_list)
                quant(:,ii) = max(quant(:,ii-1)+epsvec,quant(:,ii));
            end
            %    the above implementation isn't 100% technically correct
            %    since we could potentially end up with 
            %    quant(:,end)=onep(X)+size(X,2)*eps,
            %    whereas we want to enforce quant(:,end) <= onep(X)
            %    so, do the above process but in reverse
            quant(:,end) = min(obj.onep(X)-epsvec,quant(:,end));
            for ii=size(quant,2):-1:2
                quant(:,ii-1) = min(quant(:,ii)-epsvec,quant(:,ii-1));
            end
            assert(all(epsvec>0))
            assert(all(obj.zerop(X)<obj.onep(X)))
            assert(all(all(quant(:,1:end-1)<quant(:,2:end))))
            assert(all(obj.zerop(X)<quant(:,1)))
            assert(all(quant(:,end)<obj.onep(X)))
            % 
            % now we need to add the tails on 
            % here's some math ---
            % select ii and let f(y) be the pdf of X(ii,:), and set
            % t0=0, t1=tau_list(1), t2=tau_list(2),
            % q0=zerop(X(ii,:)), q1=quant(ii,1), q2=quant(ii,2)
            % since we do linear piecewise interpolation we know that over
            % q1<=y<=q2 that 
            % 1. f(y)=constant=a 
            % 2. \int_{q1}^{q2} f(y)dy = t2-t1
            % which together imply
            %  a*(q2-q1) = (t2-t1)  -->  a = (t2-t1)/(q2-q1)
            % now we want to figure out what f(y) is over q0<=y<=q1
            % we know (or rather, want, in 3.)
            % 3. f(y) is exponential
            % 4. \int_{q0}^{q1} f(y)dy = t1-0 = t1
            % we also want that f is continuous at y=q1, so this means that
            % we can write f(y) as
            % 5. f(y) = a*exp(-c(q1-y)),
            % where a=(t2-t1)/(q2-q1) from before and c is some constant
            % (and note that c should be positive)
            % we can combine 4. and 5. to get that
            % t1 = \int_{q0}^{q1} a*exp(-c*(q1-y)) dy =
            %               (a/c)*(1-exp(-c*(q1-q0))
            % now, letting z = c*(q1-q0) we get that
            % t1 = ((q1-q0)*a/z)*(1-exp(-z))
            % which implies
            % (1-exp(-z))/z = t1/a/(q1-q0) = (t1/(t2-t1))*((q2-q1)/(q2-q0))
            % so, we now want to solve for z in order to solve for c
            % but, since c depends on ii, this would require solving a
            % transcendental equation for each ii.  that'd be bad. instead,
            % because (1-exp(-z))/z in montone decreasing we can do a
            % little trick of calculating (1-exp(-z))/z for each z in
            % 0.00001:small_step:large_number and then use linear
            % interpolation
            % Note that we don't care about the case when w<0, since this
            % would mean the pdf increases as y goes from q1 to q0, which
            % we don't want.  Instead, in the cases where z<0 we'll just
            % set f(y) = constant and solve for the constant.
            % do the front tail
            tail0 = zeros(size(X,1),ntail); % tail on the front
            tail0(:,1)   = obj.zerop(X);
            q0 = tail0(:,1);
            q1 = quant(:,1);
            q2 = quant(:,2);
            t1 = obj.tau_list(1);
            t2 = obj.tau_list(2);
            a  = (t2-t1)./(q2-q1);
            a2 = (t1-0 )./(q1-q0); % if f(y) is contstant over q0<=y<=q1
            idxlin = a2>=a; % indexes where f(y) is linear in the tail
            idxexp = ~idxlin; % indexes where f(y) is exponential
            % now we need to solve for z for each each ii in idxexp using
            % the linear interpolation trick
            zstep = [0.00001:1:10,20:10:100,200:100:1000,2e3:1e3:1e4,2e4:1e4:1e5];
            zvals = (1-exp(-zstep))./zstep;
            if sum(idxexp)>0
                z = interp1(zvals,zstep,t1./(a(idxexp).*(q1(idxexp)-q0(idxexp))),'linear',1e5);
                c = z./(q1(idxexp)-q0(idxexp));
                assert(all(c>=0))
            end
            % now that we have c (or a2, if f is constant), we need to
            % figure out where the quantiles are for a given tau 0<t<t1
            % this reduces to finding q such that \int_{q0}^q f(y)dy = t
            % if f(y) = a2 is constant, then q(t)=q0+t/a2
            % if f(y) = a*exp(-c(q1-y)) then doing the integration out we
            % get q(t) = log( tc/a+exp(c(q0-q1)) )/c + q1
            % 
            taus0 = 0:t1/ntail:t1-t1/ntail/2;
            if sum(idxlin)>0
                tail0(idxlin,2:end) = q0(idxlin)*ones(1,ntail-1) + ...
                       (ones(sum(idxlin),1)*taus0(2:end))./(a2(idxlin)*ones(1,ntail-1));
            end
            if sum(idxexp)>0
                tail0(idxexp,2:end) = log( (ones(sum(idxexp),1)*taus0(2:end)).*(c*ones(1,ntail-1))./(a(idxexp)*ones(1,ntail-1)) + ...
                        exp(c.*(q0(idxexp)-q1(idxexp)))*ones(1,ntail-1) )./(c*ones(1,ntail-1)) + ...
                        q1(idxexp)*ones(1,ntail-1);
            end
            % now do the same thing for the other tail
            tail1 = zeros(size(X,1),ntail); % tail on the back
%             tail1(:,end) = max(obj.onep(X),quant(:,end)+eps);
            tail1(:,end) = obj.onep(X);
            q0 = tail1(:,end);
            q1 = quant(:,end);
            q2 = quant(:,end-1);
            t1 = obj.tau_list(end);
            t2 = obj.tau_list(end-1);
            a  = (t2-t1)./(q2-q1);
            a2 = (t1-1 )./(q1-q0); % if f(y) is contstant over q0<=y<=q1
            idxlin = a2>=a; % indexes where f(y) is linear in the tail
            idxexp = ~idxlin; % indexes where f(y) is exponential
            if sum(idxexp)>0
                z = interp1(zvals,zstep,(t1-1)./(a(idxexp).*(q1(idxexp)-q0(idxexp))),'linear',1e5);
                assert(all(z>=0))
                c = -z./(q1(idxexp)-q0(idxexp)); 
                assert(all(c>=0))
            end
            % the integration is different now
            % if f(y)=a2 is constant, then q(t)=q0-(1-t)/a2
            % if f(y) = a*exp(-c(y-q1)) then we get
            % q(t) = q1 - log(1+(c/a)*(t1-t))/c
            taus1 = t1+(1-t1)/ntail:(1-t1)/ntail:1;
            if sum(idxlin)>0
                
                tail1(idxlin,1:end-1) = q0(idxlin)*ones(1,ntail-1) - ...
                       (ones(sum(idxlin),1)*(1-taus1(1:end-1)))./(a2(idxlin)*ones(1,ntail-1));
            end
            if sum(idxexp)>0
                tail1(idxexp,1:end-1) = -log( 1 + (c*ones(1,ntail-1)).*( ones(sum(idxexp),1)*(t1-taus1(1:end-1)) )./(a(idxexp)*ones(1,ntail-1)))./(c*ones(1,ntail-1)) + ...
                    q1(idxexp)*ones(1,ntail-1);
            end

            % now put everything together
            taus = [taus0,obj.tau_list,taus1];
            quant = [tail0,quant,tail1]; 
            
            assert(all(all(tail1<=obj.onep(X)*ones(1,size(tail1,2)))))
        end
    end
    methods (Access=private)
        function Xtransform = transformX(obj,X)
            % this transforms X into a new space by adding more columns
            if strcmp(obj.method,'linear')
                Xtransform = X;
            elseif strcmp(obj.method,'c')
                % c: constant
                % need to add a constant 
                Xtransform = [X,ones(size(X,1),1)];
            elseif strcmp(obj.method,'cq0X')
                % c: constant
                % q0x: quadratic with no cross terms
                Xtransform = [X,ones(size(X,1),1),X.^2];
            elseif strcmp(obj.method,'QRB')
                Xtransform = X;
            else
                error('unrecognized QUANTILE REGRESSION method')
            end
        end
    end
    methods
        
        function test(obj)
            % do a visual inspection
            m = {'linear','c','cq0X'};
            xmax = 2;
            for N=[100 1000]
                [x,y,zp,op] = obj.test_generate_samples_1d(xmax,N);
                obj.input_zero_one_percentiles(zp,op);
                for ii=1:length(m)
                    obj.set_method(m{ii}); 
                    obj.clear_train();
                    obj.clear_data();
                    obj.input_data(x,y,false);
                    obj.train();
                    obj.plot_quantiles_1d();
                    title(strcat('method=',m{ii},', N=',num2str(N)))
                end
                
            end
        end
    end
    
    
end