classdef CR2R_TRAIL < CONVERT_RESP2REAL
    % the response variable is the error between the wind realization and
    % the forecast 'trail' hours previously
    properties
        trail
        do_error
    end
    methods
        function obj = CR2R_TRAIL(trail,do_error,offset)
            if nargin<2
                do_error = true;
            end
            if nargin<3
                offset = 0;
            end
            obj = obj@CONVERT_RESP2REAL(offset);
            if do_error
                do_error = 1;
            else
                do_error = 0;
            end
            obj.trail = trail;
            obj.do_error = do_error;
            obj.num_var = 1;
            obj.real_nts_lead  = trail+1;
            obj.real_nts_trail = 0;
        end
        function realvar = make_variables_fcn(obj,hist_data,hist_row,...
            respscenprev,scenrowdate,respscencur)
            % the real-world variable is just the response variable plus
            % the corresponding forecast
            % hist_data = [realization, forecast]
            f = hist_data(hist_row-obj.trail+1,obj.trail+1);
            realvar = respscencur+f*obj.do_error;
        end
    end
end