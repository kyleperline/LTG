classdef JD_GCTAEF < JOINT_DISTRIBUTION
    % Gaussian copula time adaptive exponential forgetting covariance
    properties
        lambda % forgetting factor
        sigma  % covariance
    end
    methods
        function obj = JD_GCTAEF()
            obj.train_clear();
        end
        function obj = input_lambda(obj,lambda)
            obj.lambda = lambda;
        end
        
        function obj = train_clear(obj)
            obj.sigma = double.empty(0);
        end
        function obj = train(obj, prev_MD_training) 
            obj.train_MD(prev_MD_training);
            if isempty(obj.sigma)
                obj.sigma = eye(length(obj.MDarray));
            end
            p = obj.get_percentiles_trained_last();
            z = norminv(p);
            for row=1:size(p,1)
                if all(~isnan(p(row,:)))
                    obj.sigma = obj.lambda*obj.sigma + ...
                                    (1-obj.lambda)* (z(row,:)')*z(row,:);
                end             
            end
            % recalibrate sigma
            s = diag(obj.sigma).^0.5;
            obj.sigma = obj.sigma./(s*(s'));

        end
        function s = generate(obj,margcellX,obs)
            disp('in JD_GCTAEF')
            
            % get the number of scenarios
            N = size(margcellX{1},1);
            % generate N multivariate normal random variables
            
            s = obj.sigma;
            z = mvnrnd(zeros(length(obj.MDarray),1),s,N);
            % get the percentiles of z
            p = normcdf(z,0,1);
            % get the inverse cdf of these scenarios
            s = obj.inverse_cdf(margcellX,p);
            
            N2 = 20;
            tot = zeros(1,N2+1);
            pis = zeros(1,length(margcellX));
            myranks = zeros(1,N2+1);
            myranks1 = zeros(1,5);
            for ii=1:length(margcellX)
                MDii = obj.MDarray{ii};
                X = margcellX{ii};
                pii = MDii.cdf(X(1),obs(ii));
                if ii<170
                    tot = tot+binopdf(0:N2,N2,pii);
                end
                pis(ii) = pii;
                % do rank histogram
                for jj=1:floor(size(s,1)/N2)
                   sij = sort([s(((jj-1)*N2+1):jj*N2,ii);obs(ii)]);
                   inds = find(abs(sij-obs(ii))<1e-10);
                   if length(inds)>1
                       ind = randsample(inds,1);
                   else
                       ind = inds;
                   end
                   if ii<170
                       myranks(ind) = myranks(ind)+1;
                   end
%                    if ii==1
%                        [sij';sij'<=obs(ii);(abs(sij-obs(ii))<1e-10)']
%                        find(abs(sij-obs(ii))<1e-10)
%                        [(jj-1)*N2+1,jj*N2,obs(ii),ind]
%                        myranks1(jj) = ind;
%                    end
                end
            end
%             size(tot)
            figure
            hold all
            plot(tot*floor(size(s,1)/N2))
            plot(myranks)
            legend('theo','sampled')
%             size(pis)
            
%             figure
%             hold all
%             plot(s(:,1),'*-')
%             plot([1,size(s,1)],obs(1)*[1,1])
%             disp('myranks1:')
%             myranks1
            
            
%             figure
%             hold all
%             plot(pis)
%             sa = sum( s <= ones(size(s,1),1)*obs' )/size(s,1);
%             plot(sa,'*-')
%             pa = sum( p <= ones(size(p,1),1)*pis )/size(p,1);
%             plot(pa)
%             legend('cdf','sampled inv','sampled')
            
%             figure
%             hold on
%             for ii=1:1%length(margcellX)
%                 plot(p(:,ii))
%                 runstest(p(:,ii))
% %                 plot(ii,p(:,ii),'.')
%             end
            
            
            
%             for ii=1:3
%                 MDii = obj.MDarray{ii};
%                 MDii.plot_pcdf_2d();
%             end
            
%             a=asdf;
        end
        function s = generate_conditional(obj,margcellX,respscen)
            % get the number of scenarios
            N = size(margcellX{1},1);
            % check the size of respscen
            assert(size(respscen,1)==N,'number of scenarios in predictor variables and conditioned response scenarios must be the same')
            assert(size(respscen,2)>0)
            M = length(margcellX);
            R = size(respscen,2);
            assert(R<M)
            % need to draw N (M-R)-dimensional mvn 
            % figure out new mean and covariance
            s11 = obj.sigma(1:R,1:R);
            s22 = obj.sigma(R+1:end,R+1:end);
            s12 = obj.sigma(R+1:end,1:R);
            mu  = s12/s11*(respscen');
            s   = s22-s12/s11*(s12');
            % generate N multivariate normal random variables
            z = mvnrnd(mu',s);
            % get the percentiles of z
            p = normcdf(z,0,1);
            % get the inverse cdf of these scenarios
            s = obj.inverse_cdf(margcellX,[zeros(N,R),p]);
            s = s(:,R+1:end);
        end
    end
    methods (Static)
        function [JD,p] = test_1(M,N)
            JD = JD_GCTAEF();
            d = 1;
            scale = 10;
            [margcellXy, zerop,onep,sig] = ...
                     JD.test_generate_data_1(M,N,d,scale);
            MDarray = cell(1,M);
            update  = cell(1,M);
            for ii=1:M
                qr = MD_QUANTILE_REGRESSION('c');
                qr.input_zero_one_percentiles(zerop,onep);
                MDarray{ii} = qr;
                update{ii} = false;
            end
            JD.input_lambda(0.99);
            JD.input_MDarray(MDarray);
            JD.input_data(margcellXy,update);
            JD.train();
            disp('real covariance matrix:')
            disp(sig);
            disp('estimated covariance matrix:')
            disp(JD.sigma);
            % generate 50 scenarios by the trained method and 50 scenarios
            % by the true (sig) method given one set of predictor variables
            figure()
            hold on
            ns = 10000;
            x = rand(1,M)*scale;
            ytrue = mvnrnd(zeros(1,M),sig,ns).*(1+.2*ones(ns,1)*x)+ones(ns,1)*x;
            mcX = cell(1,M);
            for ii=1:M
                mcX{ii} = x(ii)*ones(ns,1);
            end
            ytrained = JD.generate(mcX);
            plot(1:M,x,'*-k');
            nshow = 25;
            plot(ytrue(1,:),'r')
            plot(ytrained(1,:),'b');
            plot(ytrue(1:nshow,:)','r');
            plot(ytrained(1:nshow,:)','b');
            legend('mean','true','trained')
            disp('true marginal mean (top) and variance (bottom):')
            [ mean(ytrue,1); var(ytrue,1) ]
            disp('marginal distribution mean (top) and variance (bottom:')
            [ mean(ytrained,1); var(ytrained,1) ]
            p = zeros(N,M);
            for ii=1:M
                MD = JD.MDarray{ii};
                p(:,ii) = MD.cdf(MD.X,MD.y);
            end
        end
    end
    
end