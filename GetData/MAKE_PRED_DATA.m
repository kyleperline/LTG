classdef MAKE_PRED_DATA
    % a class to convert historical data into marginal distribution 
    % predictor variables
    properties
        pred_nts_before
        pred_nts_after
    end
    methods
        function obj = MAKE_MARG_DATA(pred_nts_before,pred_nts_after)
            obj.pred_nts_before = pred_nts_before;
            obj.pred_nts_after  = pred_nts_after;
        end
        function [pred_nts_before,pred_nts_after] = ...
                           get_nts_before_after(obj)
            pred_nts_before = obj.pred_nts_before;
            pred_nts_after  = obj.pred_nts_after;
        end
    end
    methods (Abstract)
        X = make_pred(obj,hist_data,row_start,row_end)
            % make the predictor variables from the historical data
            % inputs:
            %  - hist_data: NXM_{input} historical data
            %               Each row is the data from a time step
            %  - row_start: positive integer, 
            %               pbj.pred_nst_before+1<=row_start<=row_end
            %  - row_end  : positive integer,
            %               row_start<=row_end<=N-obj.pred_nts_after
            % returns:
            %  - X: (row_end-row_start+1)XM array of predictor variables
            %       X(row,:) are the predictor variables for a marginal
            %       distribution corresponding to the time step in
            %       hist_data(row+row_start-1,:)
    end
    
end