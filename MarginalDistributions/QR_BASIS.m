classdef QR_BASIS < handle
    properties
        p
    end
    methods
        function obj = QR_BASIS()
        end
        function p = get_p(obj)
            p = obj.p;
        end
    end
    methods (Abstract)
        [bfhandle, dbfhandle] = get_basis_function_handles(obj)
            % returns a function handle that outputs the quantile given
            % parameters
            % inputs:
            %  - none
            % returns:
            %  - bfhandle: a function handle as follows:
            %    y = bfhandle(params,beta)
            %    inputs:
            %     - params: NX(M_2) array of parameters
            %     - beta: pX1 array of quantile parameters
            %    returns:
            %     - y: NX1 array of quantiles
            %  - dbfhandle: a function handle as follows:
            %    J = dbfhandle(params,beta)
            %    inputs:
            %     - params: NX(M_2) array of parameters
            %     - beta: pX1 array of quantile parameters
            %    returns:
            %     - J: NX(M_2) array. J(ii,jj) is partial derivative of
            %          y(ii) with respect to beta(jj)
        binit = get_binit(obj)
            % return a first guess for beta
            % inputs: None
            % returns:
            %  - binit: pX1 array of initial guess for beta
    end
end