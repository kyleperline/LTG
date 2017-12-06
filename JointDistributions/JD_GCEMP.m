classdef JD_GCEMP < JD_GC_GENERATE
    % Gaussian copula with empirical covariance
    % NOT time adaptive
    properties
    end
    methods
        function obj = JD_GCEMP()
        end
        function obj = train(obj, prev_MD_training)
            obj.train_MD(prev_MD_training);
            p = obj.get_percentiles_all();
            myeps = 1e-6;
            p = max(min(p,1-myeps),myeps);
            z = norminv(p);
            save('tmp.mat','p','z')
            zm = ones(size(z,1),1)*mean(z);
            obj.sigma = (z-zm)'*(z-zm);
            % recalibrate sigma
            s = diag(obj.sigma).^0.5;
            obj.sigma = obj.sigma./(s*(s'));
        end
        function obj = train_clear(obj)
            obj.sigma = [];
        end
    end
    
    methods (Static)
        function d= get_description()
            d = 'JD_GCEMP';
        end
    end
end