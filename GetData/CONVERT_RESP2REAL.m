classdef CONVERT_RESP2REAL < handle
    % a class that converts response variables to real-world variables
    properties (Access=protected)
        num_var % number of real-world variables
        real_nts_lead  % >=0
        real_nts_trail % >=0
        offset
        % see MAKE_PREDRESP_DATA for fill_data, nadd
        fill_data
        nadd
    end
    methods 
        function obj = CONVERT_RESP2REAL(offset)
            obj.offset = offset;
            obj.fill_data = false;
        end
        function obj = set_fill_data(obj,fill_data)
            obj.nadd = 10;
            obj.fill_data = fill_data;
        end
        function n = get_num_vars(obj)
            n = obj.num_var;
        end
        function [real_nts_lead, real_nts_trail] = get_nts_lead_trail(obj)
            real_nts_lead  = max(0,obj.real_nts_lead -obj.offset);
            real_nts_trail = max(0,obj.real_nts_trail+obj.offset);
            if obj.fill_data
                real_nts_lead = real_nts_lead+obj.nadd;
            end
        end
        function realvar = make_variables(obj,hist_data,hist_row,...
                respscenprev,scenrowdate,respscencur)
            hist_row = hist_row+obj.offset;
            if obj.fill_data
                hist_data = obj.fill_hist_data(hist_data);
                hist_row  = obj.fill_hist_data_fix_row(hist_row);
            end
            realvar = obj.make_variables_fcn(hist_data,hist_row,...
                respscenprev,scenrowdate,respscencur);
        end
        
        function hist_data = fill_hist_data(obj,hist_data)
            % hist_data is NXM, hist_data(:,1) is realization,
            % hist_data(:,2:end) are forecasts
            r = NaN;
            f = NaN;
            for ii=1:size(hist_data,1)
                if isnan(hist_data(ii,1))
                    hist_data(ii,1) = r;
                end
                r = hist_data(ii,1);
                if isnan(hist_data(ii,2))
                    hist_data(ii,2:end) = f;
                end
                f = [hist_data(ii,3:end),hist_data(ii,end)];
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
        realvar = make_variables_fcn(obj,hist_data,hist_row,...
            respscenprev,scenrowdate,respscencur)
            % create the real-world variables 
            % inputs:
            %  - hist_data   : NXM array of historical data
            %  - hist_row    : positive integer
            %  - respscenprev: (N_2)X(M_2)XS array of S scenarios
            %                  scen(ii,:,s) is data generated in scenario 
            %                  s at time step ii 
            %  - scenrowdate :  positive integer
            %  - respscencur : SX1 array of response scenarios
            %  NOTE       : hist_row and scen_row are used to align time
            %               steps --- hist_date(hist_row+ii) is data
            %               corresponding to the same time step as data in 
            %               scen_data(scen_row+ii,:,s)
            % returns:
            %  - realvar: SX(num_var) array of real-world variables
            %             realvar(s,:) are the variables corresponding to
            %             scenario s at time step hist_row in
            %             hist_data which is time step scen_row,
            %             scen_data(scen_row,:,s)
    end
    
end