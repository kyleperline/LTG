classdef MPRD_SIDES < MAKE_PREDRESP_DATA
    % historical data is NXM array
    % first column is realization
    % columns 2:end are the forecasts
    % the predictor variables for realization in row 'row' are the
    % forecasts from n hours ago
    %
    % historical data = 
    %  [ r(1) f(1,2) f(1,3) f(1,4) ...
    %  [ r(2) f(2,3) f(2,4) f(2,5) ...
    %  [ r(3) f(3,4) f(3,5) f(3,6) ...
    % r(i)  = wind realization over time step (i-1) to i
    % f(i,j) = wind forecast created at the start of time step i that 
    %          predicts the wind realization rj
    % 
    % response variables at time step (row) i:
    %  r(j)-f(i,j)*do_error, j=i+lead_time
    %  where do_error=1 (true) or =0 (false)
    % predictor variables at time step (row) i:
    %  f(i,j), j=i+lead_time+nts_lower:i+lead_time+nts_upper
    %
    % e.g. 1 -----------------
    %
    %  lead_time = 1
    %  nts_lower = 0;
    %  nts_upper = 0;
    %  i = 3;
    %                      resp. var
    %  Xy = [ f(3,4)     , r(4)-f(3,4)*do_error    ] 
    %        pred. var, 
    %
    % e.g. 2 -----------------
    %
    %  lead_time = 2
    %  nts_lower = -1;
    %  nts_upper = 1;
    %  i = 1;
    %                                            resp. var
    %  Xy = [ f(1,2)     , f(1,3)     , f(1,4) , r(3)-f(1,3)*do_error    ] 
    %        pred. var   , pred.var, pred.var
    %
    properties
        lead_time
        nts_lower
        nts_upper
        do_error
    end
    methods
        function d = get_description(obj)
            d = ['MPRD_SIDES_',...
                    num2str(obj.lead_time),'_',...
                    num2str(obj.nts_lower),'_',...
                    num2str(obj.nts_upper),'_',...
                    num2str(obj.do_error)];
        end
        function obj = MPRD_SIDES(lead_time,nts_lower,nts_upper,do_error,offset)
            if nargin<4
                do_error = true;
            end
            if nargin<5
                offset = 0;
            end
            
            obj = obj@MAKE_PREDRESP_DATA(offset);
            assert(lead_time+nts_lower>=1);
            assert(nts_lower<=nts_upper,...
                  'lower must be no greater than upper')
            if do_error
                do_error = 1;
            else
                do_error = 0;
            end
            obj.lead_time = lead_time;
            obj.nts_lower = nts_lower;
            obj.nts_upper = nts_upper;
            obj.do_error  = do_error;
            % these three variables need to be defined from the parent
            % class
            obj.predresp_nts_lead  = 0;
            obj.predresp_nts_trail = lead_time;
            obj.pred_nts_lead      = 0;
        end
        function M = get_number_pred_vars(obj)
            M = obj.nts_upper-obj.nts_lower+1;
        end
        function [zerop,onep] = get_zero_one_percentiles(obj)
            if obj.do_error
                if obj.nts_lower<=0 && 0<=obj.nts_upper
                    n = -obj.nts_lower+1;
                    zerop = @(x)  -x(:,n);
                    onep  = @(x) 1-x(:,n);
                else
                    zerop = @(x) -ones(size(x,1),1);
                    onep  = @(x)  ones(size(x,1),1);
                end
            else
                zerop = @(x) zeros(size(x,1),1);
                onep  = @(x)  ones(size(x,1),1);
            end
                
        end
        function obj = train_fcn(obj,hist_data)
            % nothing to do
        end
        function Xy = make_predresp_fcn(obj,hist_data,row_start,row_end)
            % make the predictor and response variables from the historical
            % data
            % inputs:
            %  - hist_data: NXT matrix of historical wind data
            %               hist_data(:,1) is the wind realization
            %               hist_data(:,2:end) are the wind forecasts
            %               hist_data(ii,jj) predicts hist_data(ii+jj-1,1)
            %               for jj>1
            %  - row_start: positive integer
            %  - row_end  : positive integer
            % returns:
            %  - Xy: (row_end-row_start+1)X(nts_upper-nst_lower+2) array
            %        of predictor variables (columns 1:end-1) and response
            %        variable (column end)
            assert(row_start>=1+obj.predresp_nts_lead)
            assert(size(hist_data,1)>=row_end+obj.predresp_nts_trail)
            Xy = zeros(row_end-row_start+1,obj.nts_upper-obj.nts_lower+2);
            % the response variable is the error between the wind
            % realization and the lead_time forecast
            LT = obj.lead_time;
            Xy(:,end) = hist_data(row_start+LT:row_end+LT,1)-...
                        hist_data(row_start:row_end,LT+1)*obj.do_error;
            Xy(:,1:end-1) = ...
                 hist_data(row_start:row_end,...
                           LT+1+obj.nts_lower:LT+1+obj.nts_upper);
        end
        function [X, haspred] = make_predscen_fcn(obj,...
                hist_data,hist_row,scen_data,scen_row)
            % make the predictor variables given historical data and
            % generated scenarios
            % inputs:
            %  - hist_data: NXT matrix of historical wind data
            %               hist_data(:,1) is the wind realization
            %               hist_Data(:,2:end) are the wind forecasts
            %               hist_data(ii,jj) predicts hist_data(ii+jj-1,1)
            %               for jj>1
            %  - hist_row  : positive integer
            %  - scen_data : nX1XS 3d array
            %                scen(ii,1,s) is the generated wind scenario at
            %                time step ii in scenario s
            %  - scen_row  : positive integer
            % returns:
            %  - X: SX(nts_after+nts_before+1) array
            %       X(s,:) are the predictor variables created as in 
            %       obj.make_predresp when hist_data(1:n,1) is set to
            %       scen(:,1,s)
            %       Because the predictor variables are independent of the
            %       wind realization/scenario X(ii,:)=X(jj,:)
            %  - haspred: boolean, true if all predictor variables could be
            %  created
            LT = obj.lead_time;
            X = hist_data(hist_row,LT+1+obj.nts_lower:LT+1+obj.nts_upper);
            haspred = all(~isnan(X));
            X = ones(size(scen_data,3),1)*X;
        end
    end
    methods (Static)
        function test()
            hist_data = [1  2  4  8;
                         3  5  9  17;
                         3  7  15 31;
                         9  17 33 65;
                         16 32 64 99];
            ms = MPRD_SIDES(2,-1,1);
            pr = [2 4  8  -1;
                  5 9  17  0;
                  7 15 31  1];
            Xy = ms.make_predresp(hist_data,1,3);
            assert(all(all(pr==Xy)));
            
            ms = MPRD_SIDES(2,-1,1,true,-1);
            Xy = ms.make_predresp(hist_data,2,4);
            assert(all(all(pr==Xy)));
            
            ms = MPRD_SIDES(1,0,1,false,-1);
            pr = [2  4  3;
                  5  9  3;
                  7  15 9;
                  17 33 16];
            Xy = ms.make_predresp(hist_data,2,5);
            assert(all(all(pr==Xy)));
            disp('test passed')
        end
    end
    
end