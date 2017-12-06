classdef JD_GC_GENERATE < JOINT_DISTRIBUTION
    % implements the generation methods when using a gaussian copula
    % approach
    properties
        sigma % covariance matrix
    end
    methods
        function s = generate(obj,margcellX,numdist)
            % get the number of scenarios
            N = size(margcellX{1},1);
            % generate N multivariate normal random variables
            s = obj.sigma;
            s = s(1:numdist,1:numdist);
            z = mvnrnd(zeros(numdist,1),s,N);
            % get the percentiles of z
            p = normcdf(z,0,1);
            % get the inverse cdf of these scenarios
            s = obj.inverse_cdf(margcellX(1:numdist),p);
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
    
end