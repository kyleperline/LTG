classdef QRBF_LINEAR < QR_BASISFUNC
    properties
    end
    methods
        function obj = QRBF_LINEAR()
            obj.p = 1;
        end
        function binit = get_binit(obj)
            binit = 0.5;
        end
        function y = basis(obj,param,beta)
            y = param*beta;
        end
        function dydb = dbasis(obj,param,beta)
            dydb = param;
        end
    end
end