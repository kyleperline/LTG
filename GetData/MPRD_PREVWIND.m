classdef MPRD_PREVWIND < MAKE_PREDRESP_DATA
    % historical data = 
    %  [ r1 f12 f13 f14 ...
    %  [ r2 f23 f24 f25 ...
    %  [ r3 f34 f35 f36 ...
    % ri  = wind realization over time step (i-1) to i
    % fij = wind forecast created at the start of time step i the predicts
    %       the wind realization rj
    % 
    % response variables at time step (row) i:
    %  rj-fij*do_error, j=i+1
    %  where do_error=1 (true) or =0 (false)
    % predictor variables at time step (row) i:
    %  rk-fij, j=i+1, k=i-nprev_lower+1:i-nprev_upper+1
    %
    % e.g.
    %
    %  nprev_lower = 3;
    %  nprev_upper = 2;
    %  i = 3;
    %  e = do_error;
    %                               resp. var
    %  Xy = [ r1-f34*e  , r2-f34*e  , r4-f34*e   ] 
    %        pred. var  , pred.var
    % 
    %
    properties
        nprev_lower
        nprev_upper
        do_error
    end
    methods
        function obj = MPRD_PREVWIND(nprev_lower,nprev_upper,do_error)
            assert(nprev_lower>=nprev_upper)
            assert(nprev_upper>0)
            if nargin==2
                do_error = true;
            end
            if do_error
                do_error = 1;
            else
                do_error = 0;
            end
            obj.nprev_lower = nprev_lower;
            obj.nprev_upper = nprev_upper;
            obj.do_error    = do_error;
            % these three variables need to be defined from the parent
            % class
            obj.predresp_nts_lead  = nprev_lower;
            obj.predresp_nts_trail = 1;
            obj.pred_nts_lead      = nprev_lower;
        end
        function M = get_number_pred_vars(obj)
            M = obj.nprev_lower-obj.nprev_upper+1;
        end
        function Xy = make_predresp(obj,hist_data,row_start,row_end)
            Xy = zeros(row_end-row_start+1,...
                    obj.nprev_lower-obj.nprev_upper+2);
            Xy(:,end) = hist_data(row_start+1:row_end+1,1)-...
                hist_data(row_start:row_end,2)*obj.do_error;
            for ii=1:size(Xy,2)-1
                Xy(:,ii) = hist_data(row_start+1-ii:row_end+1-ii,1)-...
                    hist_data(row_start:row_end,2)*obj.do_error;
            end
        end
        function [X, haspred] = make_predscen(obj,...
                         hist_data,hist_row,scen_data,scen_row)
            % if scen_row-obj.nprev_lower<=0 then there's not enough time
            % steps in the scenarios
            % the additional time steps are copied from historical data
            if scen_row-obj.nprev_lower+1<=0
                [~,nc,ns] = size(scen_data);
                sd2 = zeros(1-(scen_row-obj.nprev_lower+1),nc,ns);
                % each row in sd2 is historical data at corresponding time
                % step
                offset = hist_row-obj.nprev_lower;
                for row=1:size(sd2,1)
                    sd2(row,:,:) = hist_data(row+offset,1);
                end
                scen_data = cat(1,sd2,scen_data);
                scen_row  = scen_row+size(sd2,1);
            end
            X = scen_data(scen_row-obj.nprev_lower+1:scen_row-obj.nprev_upper+1,1,:);
            X = reshape(X,obj.nprev_lower-obj.nprev_upper+1,...
                          size(scen_data,3),1);
            X = X - hist_data(hist_row,2)*obj.do_error;
            X = fliplr(X');
            haspred = ~any(isnan(X));
        end
        function [zerop,onep] = ...
                get_zero_one_percentiles(obj)
            if obj.do_error
                zerop = @(x) -ones(size(x,1),1);
                onep  = @(x)  ones(size(x,1),1);
            else
                zerop = @(x) zeros(size(x,1),1);
                onep  = @(x)  ones(size(x,1),1);
            end
%             ztightbd = false;
%             otightbd = false;
        end
    end
    methods (Static)
        
        function test()
            hist_data = [1  2  4  8;
                         4  5  9  17;
                         3  7  15 31;
                         9  17 33 65;
                         16 32 64 99];
           mp = MPRD_PREVWIND(2,1);
           Xy = mp.make_predresp(hist_data,2,4);
           a  = [ 4-5, 1-5, 3-5; 
                  3-7, 4-7, 9-7;
                  9-17, 3-17, 16-17];
           assert(all(all(a==Xy)))
           scen_data = zeros(4,1,3);
           scen_data(:,:,1) = [ 1 ; 2; 3; 4];
           scen_data(:,:,2) = [ 1.5;2.5;3.5;4.5];
           scen_data(:,:,3) = [ -1;-2;-3;-4];
           X = mp.make_predscen(hist_data,4,scen_data,2);
           a = [2 1;
                2.5 1.5;
                -2 -1]-17;
%            a = [ 1   2;
%                  1.5 2.5;
%                  -1  -2 ]-17;
           assert(all(all(a==X)));
           disp('test passed')
%            hist_row = 4;
%            hist_data(1:hist_row,:)
%            scen_data(1:1,:,:)
%            mp.make_predscen(hist_data,hist_row,scen_data,1)
        end
    end
end