classdef QRBF_SIG < QR_BASISFUNC
    properties
        constant_b
        bval
    end
    methods
        function obj = QRBF_SIG()
            obj.constant_b = [false, false, false];
            obj.bval = [0,0,0];
            obj.p = 3;
        end
        function obj = make_constant_b(obj,ind)
            if ind==1
                if ~obj.constant_b(1)
                    obj.p = obj.p-1;
                end
                obj.constant_b(1) = true;
            elseif ind==2
                if ~obj.constant_b(2)
                    obj.p = obj.p-1;
                end
                obj.constant_b(2) = true;
            else
                if ~obj.constant_b(3)
                    obj.p = obj.p-1;
                end
                obj.constant_b(3) = true;
            end
        end
        function obj = set_constant_bval(obj,ind,val)
            if ind==1
                if ~obj.constant_b(1)
                    error('first call make_constant_b')
                end
                obj.bval(1) = val;
            elseif ind==2
                if ~obj.constant_b(2)
                    error('first call make_constant_b')
                end
                obj.bval(2) = val;
            else
                if ~obj.constant_b(3)
                    error('first call make_constant_b')
                end
                obj.bval(3) = val;
            end
        end
        function binit = get_binit(obj)
            binit = 0.5;
        end
        function y = basis(obj,param,beta)
            beta = obj.get_beta(beta);
            y = atan(beta(1)*(param-beta(2))).*beta(3);
        end
        function dydb = dbasis(obj,param,beta)
            beta = obj.get_beta(beta);
            dydb = zeros(size(param,1),3);
            dydb(:,1) = beta(3)*(param-beta(2))./( (beta(1)*(param-beta(2))).^2+1 );
            dydb(:,2) = beta(3)*beta(1)./( (beta(1)*(param-beta(2))).^2+1 );
            dydb(:,3) = atan(beta(1)*(param-beta(2)));
            dydb = dydb(:,~obj.constant_b);
        end
    end
    methods (Access=private)
        function beta = get_beta(obj,beta)
            bt = zeros(3,1);
            ct = 1;
            if obj.constant_b(1)
                bt(1) = obj.bval(1);
            else
                bt(1) = beta(ct);
                ct = ct+1;
            end
            if obj.constant_b(2)
                bt(2) = obj.bval(2);
            else
                bt(2) = beta(ct);
                ct = ct+1;
            end
            if obj.constant_b(3)
                bt(3) = obj.bval(3);
            else
                bt(3) = beta(ct);
            end
            beta = bt;
        end
    end
end