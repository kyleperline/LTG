classdef MPRD_F_FOLLOW < MAKE_PREDRESP_DATA
    % this requires custom data
    % the forecast predictor variable follows a wind power sequence
    %
    % historical data = 
    % [ r f c ]
    % where 
    %  r = mX1 array of wind power realization as usual
    %  f = mX72 array of wind power forecasts as usual
    %  c = mX1 array of custom data
    %
    % response variable at time step (row) i:
    %  f(i,i+j), j=horizon (where f(i,j) is the forecast at time i predicting
    %                     the wind power at time j)
    % predictor variable at time step (row) i:
    %  c(i+j+shift)
    %
    % e.g. -------------
    %  horizon = 3
    %  shift = 0
    %  i = 2
    %                    resp. var
    % Xy = [ c(5)     ,  f(3,5)    ]
    %        pred. var
    %
    
    properties
        horizon
        shift
    end
    methods
        function d = get_description(obj)
            d = ['MPRD_F_FOLLOW',...
                    num2str(obj.horizon),'_',...
                    num2str(obj.shift)];
        end
        function obj = MPRD_F_FOLLOW(horizon,shift,offset)
            obj = obj@MAKE_PREDRESP_DATA(offset);
            obj.horizon = horizon;
            obj.shift = shift;
            % these three variables need to be defined from the parent
            % class
            obj.predresp_nts_lead  = 0;
            obj.predresp_nts_trail = horizon+shift;
            obj.pred_nts_lead      = 0;
        end
        function M = get_number_pred_vars(obj)
            M = 1;
        end
        function [zerop,onep] = get_zero_one_percentiles(obj)
            zerop = @(x) zeros(size(x,1),1);
            onep  = @(x)  ones(size(x,1),1);
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
            %               hist_data(:,2:73) are the wind forecasts
            %                hist_data(ii,jj) predicts hist_data(ii+jj-1,1)
            %                for jj>1
            %               hist_data(:,74) is the custom data
            %  - row_start: positive integer
            %  - row_end  : positive integer
            % returns:
            %  - Xy: (row_end-row_start+1)X(nts_upper-nst_lower+2) array
            %        of predictor variables (columns 1:end-1) and response
            %        variable (column end)
            assert(row_start>=1+obj.predresp_nts_lead)
            assert(size(hist_data,1)>=row_end+obj.predresp_nts_trail)
            Xy = zeros(row_end-row_start+1,2);
            % predictor variables
            tmp = obj.horizon + obj.shift-1;
            Xy(:,1) = hist_data(row_start+tmp:row_end+tmp,74);
            % reponse variable
            Xy(:,2) = hist_data(row_start:row_end,1+obj.horizon);
        end
        function [X, haspred] = make_predscen_fcn(obj,...
                hist_data,hist_row,scen_data,scen_row)
            % make the predictor variables given historical data and
            % generated scenarios
            % inputs:
            %  - hist_data: NXT matrix of historical wind data
            %               hist_data(:,1) is the wind realization
            %                hist_data(:,2:73) are the wind forecasts
            %                hist_data(ii,jj) predicts hist_data(ii+jj-1,1)
            %                for jj>1
            %               hist_data(:,74) is the custom data
            %  - hist_row  : positive integer
            %  - scen_data : nXHXS 3d array
            %                scen(ii,:,s) is the generated wind forecast at
            %                time step ii in scenario s
            %  - scen_row  : positive integer
            % returns:
            %  - X: SXM array of predictor variables
            %       X(s,:) are the predictor variables corresponding to
            %       time step hist_row in hist_data which is time step
            %       scen_row in scenario s, scen_data(scen_row,:,s)
            %  - haspred: boolean, true if all predictor variables could be
            %  created
            
            % the predictor variables are independent of the forecast
            % scenarios
            X = hist_data(hist_row+obj.horizon+obj.shift);
            haspred = ~isnan(X);
            X = ones(size(scen_data,3),1)*X;
        end
    end
    
    
end