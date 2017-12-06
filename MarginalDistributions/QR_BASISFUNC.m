classdef QR_BASISFUNC < handle
    properties
        p
    end
    methods
        function obj = QR_BASISFUNC()
        end
        function p = get_p(obj)
            p = obj.p;
        end
    end
    methods (Abstract)
        binit = get_binit(obj)
        y = basis(obj,param,beta)
        dydb = dbasis(obj,param,beta)
    end
end