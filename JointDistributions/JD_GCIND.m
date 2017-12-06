classdef JD_GCIND < JOINT_DISTRIBUTION
    % independent Gaussian copula
        
    methods
        function obj = JD_GCIND()
        end
        function obj = train(obj, prev_MD_training)
            obj.train_MD(prev_MD_training);
        end
        function obj = train_clear(obj)
        end
        function s = generate(obj,margcellX)
            % get the number of scenarios
            N = size(margcellX{1},1);
            % generate N multivariate normal random variables
            z = randn(N,length(obj.MDarray));
            % get the percentiles of z
            p = normcdf(z,0,1);
            % get the inverse cdf of these scenarios
            s = obj.inverse_cdf(margcellX,p);
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
            z = randn(N,M-R);
            % get the percentiles of z
            p = normcdf(z,0,1);
            % get the inverse cdf of these scenarios
            s = obj.inverse_cdf(margcellX,[zeros(N,R),p]);
            s = s(:,R+1:end);
        end
    end
    methods (Static)
        function d = get_description()
            d = 'JD_GCIND';
        end
    end
    
    
end