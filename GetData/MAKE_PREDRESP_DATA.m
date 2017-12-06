classdef MAKE_PREDRESP_DATA < handle
    % a class to convert historical data into marginal distribution 
    % predictor variables and a response variable
    properties (Access=protected)
        % creating the predictor and response variables at time step ii may
        % require data from previous or later time steps
        % creating the predictor and response varaibles at time step ii
        % require data from time steps ii-predresp_nts_lead to
        % ii+predresp_nts_trail
        % creating just the predictor variables at time step ii requires
        % data over time steps ii-pred_nts_lead to ii (note that the
        % predictor variables cannot depend upon future information, which
        % is why there is no pred_nts_trail variable)
        predresp_nts_lead  % >=0
        predresp_nts_trail % >=0
        pred_nts_lead      % >=0
        % there's some missing forecasts. this can be handled by assuming
        % persistence in the forecasts. i.e. if forecast at time step ii is
        % f=1XT array, then at time step ii+1 the forecast is
        % [f(2:end),f(end)].  There's also a couple of missing
        % realizations.  This can be handled the same way.
        fill_data % boolean on whether to fill in missing data with this method
        fill_data_train
        fill_data_pred
        nadd % need to load in more data for this fill method
        % can also offset the data
        offset
    end
    methods
        function obj = MAKE_PREDRESP_DATA(offset)
            % NOTE:
            % instantiation must initialize predresp_nts_lead, 
            % predresp_nts_trail, and pred_nts_lead
            obj.fill_data = true; % TODO: fill_data should always be equal 
                                   % to (fill_data_train | fill_data_pred)
                                   % though always being true also works
            obj.fill_data_train = false;
            obj.fill_data_pred  = true;
            obj.nadd = 10;
            obj.offset = offset;
        end
        function obj = set_fill_data_train(obj,fill_data_train)
%             obj.nadd = 10;
            obj.fill_data_train = fill_data_train;
%             if obj.fill_data && ~fill_data
%                 % change obj.fill_data from true to false
%                 obj.fill_data = fill_data;
%             end
%             if ~obj.fill_data && fill_data
%                 % change obj.fill_data from false to true
%                 obj.fill_data = fill_data;
%             end
        end
        function obj = set_fill_data_pred(obj,fill_data_pred)
            obj.fill_data_pred = fill_data_pred;
        end
        function [pr_nts_lead,pr_nts_trail,p_nts_lead,p_nts_trail] = ...
                                    get_nts_lead_trail(obj)
            pr_nts_lead  = max(0,obj.predresp_nts_lead-obj.offset);
            pr_nts_trail = max(0,obj.predresp_nts_trail+obj.offset);
            p_nts_lead   = max(0,obj.pred_nts_lead-obj.offset);
            p_nts_trail  = max(0,obj.offset);
            if obj.fill_data
                pr_nts_lead = pr_nts_lead + obj.nadd;
                p_nts_lead  = p_nts_lead  + obj.nadd;
            end
        end
        
        
        function obj = train(obj,hist_data)
            if obj.fill_data_train
                hist_data = obj.fill_hist_data(hist_data);
            end
            obj = obj.train_fcn(hist_data);
        end
        function Xy = make_predresp(obj,hist_data,row_start,row_end)
            if obj.fill_data && obj.fill_data_train
                hist_data = obj.fill_hist_data(hist_data);
                row_start = obj.fill_hist_data_fix_row(row_start);
                row_end   = obj.fill_hist_data_fix_row(row_end);
            end
            Xy = obj.make_predresp_fcn(hist_data,row_start+obj.offset,row_end+obj.offset);
        end
        function [X, haspred] = make_predscen(obj,hist_data,hist_row,scen_data,scen_row)
            if obj.fill_data && obj.fill_data_pred
                hist_data = obj.fill_hist_data(hist_data);
                hist_row  = obj.fill_hist_data_fix_row(hist_row);
            end
            [X, haspred] = obj.make_predscen_fcn(hist_data,hist_row+obj.offset,scen_data,scen_row+obj.offset);
        end
        
        function hist_data = fill_hist_data(obj,hist_data)
            % hist_data is NXM, hist_data(:,1) is realization,
            % hist_data(:,2:end) are forecasts
            r = NaN;
            f = NaN;
            forecast_horizon = 72;
            for ii=1:size(hist_data,1)
                if isnan(hist_data(ii,1))
                    hist_data(ii,1) = r;
                end
                r = hist_data(ii,1);
                if isnan(hist_data(ii,2))
                    hist_data(ii,2:forecast_horizon+1) = f;
                end
                f = [hist_data(ii,3:forecast_horizon+1),hist_data(ii,forecast_horizon+1)];
            end
            % remove the first nadd lines that were added because of this
            % fill method
            hist_data = hist_data(obj.nadd+1:end,:);
        end
        function row = fill_hist_data_fix_row(obj,row)
            % when fill_hist_data(hist_data) is called, the first few rows 
            % are removed from hist_data
            % in some functions there are input arguments specifying a row
            % of hist_data
            % this row is no longer correct since the first few rows of
            % hist_data were removed
            % so, need to correct the row number
            row = row - obj.nadd;
        end
    end
    methods (Abstract)
        [zerop,onep] = get_zero_one_percentiles(obj)
            % return function handles for the zero and one hundred
            % percentiles
        M = get_number_pred_vars(obj)
            % return the number of predictor variables
            % inputs:
            % returns:
            %  - M: positive integer, number of predictor variables
        obj = train_fcn(obj,hist_data)
            % do any training on historical data that might be required for
            % computing the predictor and response variables (e.g. changing
            % basis)
        Xy = make_predresp_fcn(obj,hist_data,row_start,row_end)
            % make the predictor and response variables from the 
            % historical data
            % inputs:
            %  - hist_data: NXM_{input} historical data
            %               Each row is the data from a time step
            %  - row_start: positive integer, 
            %               obj.predresp_nst_lead+1<=row_start<=row_end
            %  - row_end  : positive integer,
            %               row_start<=row_end<=N-obj.predresp_nts_trail
            % returns:
            %  - Xy: (row_end-row_start+1)XM array of predictor and 
            %        response variables
            %        X(row,1:end-1) are the predictor variables for a 
            %        marginal distribution corresponding to the time step 
            %        in hist_data(row+row_start-1,:); X(row,end) is the
            %        response variable
        [X, haspred] = make_predscen_fcn(obj,hist_data,hist_row,scen_data,scen_row)
            % make the predictor variables from both the historical data
            % and from generated scenarios 
            % inputs:
            %  - hist_data: (N_1)XM_{input} historical data
            %  - hist_row : positive integer,
            %               pred_nts_lead<=hist_row<=N
            %  - scen_data: (N_2)X(M_2)XS array of S scenarios
            %               scen(ii,:,s) is data generated in scenario s at 
            %               time step ii 
            %  - scen_row : positive integer
            %  NOTE       : hist_row and scen_row are used to align time
            %               steps --- hist_date(hist_row+ii) is data
            %               corresponding to the same time step as data in 
            %               scen_data(scen_row+ii,:,s)
            % returns:
            %  - X: SXM array of predictor variables
            %       X(s,:) are the predictor variables corresponding to
            %       time step hist_row in hist_data which is time step
            %       scen_row in scenario s, scen_data(scen_row,:,s)
    end
    
end