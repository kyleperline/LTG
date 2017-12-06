classdef QRB_LINCOMBO < QR_BASIS
    % a linear combination of nonlinear functions
    properties
        qrbf_cell
        binit
    end
    methods
        function obj = QRB_LINCOMBO()
        end
        function obj = input_qrbf_cell(obj,qrbf_cell)
            obj.qrbf_cell = qrbf_cell;
            obj.p = 0;
            for ii=1:length(obj.qrbf_cell)
               f = obj.qrbf_cell{ii};
               obj.p = obj.p+f.get_p();
            end
            obj.binit = zeros(obj.p,1);
            pi = 1;
            for ii=1:length(obj.qrbf_cell)
                f = obj.qrbf_cell{ii};
                dp = f.get_p();
                obj.binit(pi:pi+dp-1) = f.get_binit();
                pi = pi + dp;
            end
        end
        function [bfhandle, dbfhandle] = ...
                get_basis_function_handles(obj)
            bfhandle  = @(params,beta) obj.basis_func(params,beta);
            dbfhandle = @(params,beta) obj.dbasis_func(params,beta);
        end
        function binit = get_binit(obj)
            binit = obj.binit;
        end
    end
    methods (Access=private)
        function y =basis_func(obj,params,beta)
            pi=1;
            y = zeros(size(params,1),1);
            for ii=1:length(obj.qrbf_cell)
                f = obj.qrbf_cell{ii};
                dp = f.get_p();
                y = y + f.basis(params(:,ii),beta(pi:pi+dp-1,1));
                pi = dp+pi;
            end
        end
        function y = dbasis_func(obj,params,beta)
            pi=1;
            y = zeros(size(params,1),obj.p);
            for ii=1:length(obj.qrbf_cell);
                f = obj.qrbf_cell{ii};
                dp = f.get_p();
                y(:,pi:pi+dp-1) = f.dbasis(params(:,ii),beta(pi:pi+dp-1,1));
                pi = dp+pi;
            end
        end
    end
    methods (Static)
        function QR = test()
            % do a test with a sig and linear basis function
            
            N = 500;
            X = rand(N,2);
            
            beta = [10, 0.5, 0.4, 0.2]';
            y1 = atan(beta(1)*(X(:,1)-beta(2))).*beta(3);
            y2 = X(:,2)*beta(4);
            
%             beta = [-1, 0.5, 0.2]';
%             y1 = X(:,1)*beta(1)+beta(2);
%             y2 = X(:,2)*beta(3);
            
            y = y1+y2+(rand(N,1)*2-1)*0.3;
            qrbfcell = cell(1,2);
            sig = QRBF_SIG();
            sig.make_constant_b(1);
            sig.set_constant_bval(1,beta(1));
            sig.make_constant_b(3);
            sig.set_constant_bval(3,beta(3));
            qrbfcell{1} = sig;
            qrbfcell{2} = QRBF_AFFINE();
            qrb = QRB_LINCOMBO();
            qrb.input_qrbf_cell(qrbfcell);
            QR = MD_QUANTILE_REGRESSION('linear'); % method doesn't matter
            QR.input_qrb(qrb);  
            QR.input_data(X,y,false);
            QR.input_zero_one_percentiles(@(x) -10*ones(size(x,1),1),...
                                          @(x)  10*ones(size(x,1),1));
            QR.train()
            % plot slices
            % vary dimension 1
            dim2 = 0.5;
            delta = 0.1;
            QR.plot_quantiles_1d_slice(1,[0.5,dim2])
            close = QR.X(:,2)>dim2-delta & QR.X(:,2)<dim2+delta;
            plot(QR.X(close,1),QR.y(close),'r.')
            % vary dimension 2
            dim1 = 0.5;
            delta = 0.1;
            QR.plot_quantiles_1d_slice(2,[dim1,0.5])
            close = QR.X(:,1)>dim2-delta & QR.X(:,1)<dim2+delta;
            plot(QR.X(close,2),QR.y(close),'r.')
            
                
            
        end
    end
end