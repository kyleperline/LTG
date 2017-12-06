classdef QRBF_AFFINE < QR_BASISFUNC
    properties
    end
    methods
        function obj = QRBF_AFFINE()
            obj.p = 2;
        end
        function binit = get_binit(obj)
            binit = [0.5,0.5];
        end
        function y = basis(obj,param,beta)
            y = param*beta(1)+beta(2);
        end
        function dydb = dbasis(obj,param,beta)
            dydb = [param, ones(size(param,1),1)];
        end
    end
end