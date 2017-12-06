classdef CR2R_ID < CONVERT_RESP2REAL
    % identity
    methods
        function obj = CR2R_ID(offset)
            obj = obj@CONVERT_RESP2REAL(offset);
        end
        function realvar = make_variables_fcn(obj,hist_data,hist_row,...
                respscenprev,scenrowdate,respscencur)
            realvar = respscencur;
        end
    end
end