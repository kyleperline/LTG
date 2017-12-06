classdef GET_MARG_DATA < handle
    % this class is used to get data for a marginal distribution from
    % historical data
    properties
        DH % DATA_HANDLER instance
        window
        max_pr_nts_lead
        max_pr_nts_trail
        max_p_nts_lead
        max_p_nts_trail
        % array of MAKE_PREDRESP_DATA instances
        MPRDarray
        npredvars
        zopcell
        % array of CONVERT_RESP2REAL instances
        CR2Rarray
        max_CR2R
        max_real_nts_lead
        max_real_nts_trail
        % expanding and sliding window
        window_nts_before
        window_nts_after
        prev_date
        % constant window
        window_c_start
        window_c_end
        window_c_bool
    end
    methods
        function obj = GET_MARG_DATA()
        end
        function obj = input_MPRDarray(obj,MPRDarray)
            obj.MPRDarray = MPRDarray;
            % need to get the maximum lead and trail number of time steps
            obj.max_pr_nts_lead = 0;
            obj.max_pr_nts_trail = 0;
            obj.max_p_nts_lead = 0;
            obj.max_p_nts_trail = 0;
            obj.npredvars = zeros(1,size(MPRDarray,2));
            % also need to get the 0 and 100 percentile functions
            obj.zopcell = cell(2,size(MPRDarray,2));
            for col = 1:size(MPRDarray,2)
                pcol = 1;
                for row = 1:size(MPRDarray,1)
                    if ~isempty(MPRDarray{row,col})
                        MPRD = MPRDarray{row,col};
                        [prlead,prtrail,plead,ptrail] = MPRD.get_nts_lead_trail();
                        obj.max_pr_nts_lead  = max(prlead,obj.max_pr_nts_lead);
                        obj.max_pr_nts_trail = max(prtrail,obj.max_pr_nts_trail);
                        obj.max_p_nts_lead   = max(plead,obj.max_p_nts_lead);
                        obj.max_p_nts_trail  = max(ptrail,obj.max_p_nts_trail);
                        np = MPRD.get_number_pred_vars();
                        obj.npredvars(col) = obj.npredvars(col) + np;
                        [zp,op] = MPRD.get_zero_one_percentiles();
                        if row==1
                            zpfunc = @(x) zp(x(:,pcol:pcol+np-1));
                            opfunc  = @(x) op(x(:,pcol:pcol+np-1));
                        else
%                             zpfunc = @(x)max(zp(x(:,pcol:pcol+np-1)),zpfunc(x));
%                             opfunc = @(x)min(op(x(:,pcol:pcol+np-1)),opfunc(x));
                        end
                        pcol = pcol+np;
                    end
                end
                obj.zopcell{1,col} = zpfunc;
                obj.zopcell{2,col}  = opfunc;
            end
        end
        function obj = input_CR2Rarray(obj,CR2Rarray)
            obj.CR2Rarray = CR2Rarray;
            obj.max_CR2R = 0;
            obj.max_real_nts_lead = 0;
            obj.max_real_nts_trail = 0;
            for ii=1:length(CR2Rarray)
                CR2R = CR2Rarray{ii};
                obj.max_CR2R = max(obj.max_CR2R,CR2R.get_num_vars());
                [lead, trail] = CR2R.get_nts_lead_trail();
                obj.max_real_nts_lead = max(obj.max_real_nts_lead,lead);
                obj.max_real_nts_trail = max(obj.max_real_nts_trail,trail);
            end
        end

        function obj = input_DH(obj,DHinstance)
            % input the DATA_HANDLER instance
            obj.DH = DHinstance;
        end
        
        function obj = init_window_expand_slide(obj,...
                             window,window_nts_before,window_nts_after)
            obj.window     = window;
            obj.window_nts_before = window_nts_before;
            obj.window_nts_after  = window_nts_after;
            obj.prev_date  = NaN; % used for expanding and sliding window
            % and initialize DH
            % UPDATE: the following function isn't really needed anymore
            % (it was only implemented to save on load times, but making
            % the switch to loading .mat files instead of Excel files fixed
            % that).
            % if loading data ever becomes an issue you can uncomment this
            % line --- however, beware that GEN_SCENARIOS.generate
            % functions probably won't work
%             obj.DH.init_slide_extend_window();
        end
        function obj = init_window_constant(obj,window_start,window_end)
            obj.window = 'constant';
            obj.window_c_start = window_start;
            obj.window_c_end   = window_end;
            obj.window_c_bool  = false;
            % and initialize DH
            obj.DH.init_constant_window();
        end
        
        function zopcell = get_zero_one_percentiles(obj)
            zopcell = obj.zopcell;
        end
        function M = get_num_marg_dist(obj)
            M = size(obj.MPRDarray,2);
        end
        
        function [margcellXy, update] = get_predresp_custom(obj,start_date,end_date)
            % get the data corresponding to cur_date only
            [margcellXy, update] = ...
                obj.get_predresp_window_constant(start_date,end_date);
        end
        function [margcellXy, update] = get_predresp(obj,cur_date)
            if strcmp(obj.window,'constant')
                [margcellXy, update] = obj.get_predresp_window_constant();
            elseif strcmp(obj.window,'expand')
                [margcellXy, update] = obj.get_predresp_window_expand(cur_date);
            elseif strcmp(obj.window,'slide')
                [margcellXy, update] = obj.get_predresp_window_slide(cur_date);
            else
                error('nope')
            end
        end
        function [margcellX, haspred] = ...
                get_pred_given_resp(obj,cur_date,respscenprev,scenrowdate)
            % get the predictor variables at cur_data given response 
            % variable scenarios that were generated at previous time steps
            % inputs: 
            %  - cur_date    : datenum
            %  - respscenprev: (N_2)X(M_2)XS array of S scenarios
            %  - scenrowdate : positive integer
            %                  scen(scenrowdata,:,s) is data generated in
            %                  scenario s at the time step corresponding to
            %                  cur_date
            % returns:
            %  - margcellX: 1XM cell array of predictor variables
            %  - haspred  : boolean, true if the predictor variables were
            %               created, false otherwise
            [margcellX, haspred] = ...
                    obj.make_pred_given_resp(cur_date,respscenprev,scenrowdate);
        end
        function scenarios = get_scen_given_resp_and_scen(...
                obj,cur_date,respscenprev,scenrowdate,respscencur)
            % get the scenarios given response variable scenarios
            % inputs: 
            %  - cur_date    : datenum
            %  - respscenprev: (N_2)X(M_2)XS array of S scenarios
            %  - scenrowdate : positive integer
            %                  scen(scenrowdata,:,s) is data generated in
            %                  scenario s at the time step corresponding to
            %                  cur_date
            %  - respscencur : SXM array of response variable scenarios
            %                  generated from JOINT_DISTRIBUTION
            % returns:
            %  - scenarios: MX(M_3)XS array of real-world scenarios
            %               scenarios(ii,:,s) are the real-world variables
            %               of marginal distribution ii given response
            %               variable respscencur(s,ii) 
            scenarios = obj.make_scen_given_resp_and_scen(...
                    cur_date,respscenprev,scenrowdate,respscencur);
        end

    end
    
    methods (Static,Access=protected)
        function nts = DTS(nhours)
            nts = datenum(0,0,0,nhours,0,0);
        end
        function nts = NTS(start_date,end_date)
            nts = floor(etime(datevec(start_date),datevec(end_date))/3600)+1;
        end
    end
    methods (Access=protected)
        function [margcellXy, update] = ...
                             get_predresp_window_constant(obj,start_date,end_date)
            % window == 'constant'
            override = true;
            if nargin==1
                start_date = obj.window_c_start;
                end_date   = obj.window_c_end;
                override   = false;
            end
            update = cell(1,size(obj.MPRDarray,2));
            wcbool = obj.window_c_bool;
            if isempty(wcbool)
                wcbool = false;
            end
            if override || wcbool
                hist_data = obj.DH.get_data(...
                    start_date-obj.DTS(obj.max_pr_nts_lead),...
                    end_date   +obj.DTS(obj.max_pr_nts_trail));
                margcellXy = obj.make_predresp(hist_data);
                for ii=1:length(update)
                    update{ii} = false; 
                end
            else
                margcellXy = cell(1,size(obj.MPRDarray,2));
                % update data with no new information
                for ii=1:size(obj.MPRDarray,2)
                    margcellXy{ii} = [];
                    update{ii}     = true;
                end
            end
            obj.window_c_bool = true;
        end
        function [margcellXy, update] = ...
                             get_predresp_window_expand(obj,cur_date)
            % window == 'expand'
            u = true;
            if isnan(obj.prev_date)
                % then set prev_date = cur_date - nts_before
                obj.prev_date = cur_date-obj.DTS(obj.window_nts_before);
                u = false;
            else
                assert(cur_date+obj.DTS(obj.window_nts_after+1)>=obj.prev_date)
            end
            % get data between prev_date-hours(mdata_nts_before) and
            % cur_date+hours(mdata_nts_after+window_nts_after)
            hist_data = obj.DH.get_data(...
               obj.prev_date-obj.DTS(obj.max_pr_nts_lead),...
               cur_date+obj.DTS(obj.max_pr_nts_trail+obj.window_nts_after));
            obj.prev_date = cur_date+obj.DTS(obj.window_nts_after+1);
            margcellXy = obj.make_predresp(hist_data);
            update   = cell(1,length(margcellXy));
            for ii=1:length(margcellXy)
                update{ii} = u;  % update previous data
            end
        end
        function [margcellXy, update] = ...
                             get_predresp_window_slide(obj,cur_date)
            % window == 'slide'
            % get data between 
            % cur_date-hours(mdata_nts_before-window_nts_before) and
            % cur_date+hours(mdata_nts_after+window_nts_after)
            hist_data = obj.DH.get_data(...
              cur_date-obj.DTS(obj.max_pr_nts_lead+obj.window_nts_before),...
              cur_date+obj.DTS(obj.max_pr_nts_trail+obj.window_nts_after));
            obj.prev_date = cur_date+obj.DTS(obj.max_pr_nts_trail);
            margcellXy = obj.make_predresp(hist_data);
            update   = cell(1,length(margcellXy));
            for ii=1:length(margcellXy)
                update{ii} = false;  % replace previous data
            end
        end
        function [margcellX, haspred] = ...
                    make_pred_given_resp(obj,cur_date,scen,scenrowdate)
            % load in the necessary historical data
            hist_data = obj.DH.get_data(...
                cur_date-obj.DTS(obj.max_p_nts_lead),...
                cur_date+obj.DTS(obj.max_p_nts_trail),false);            
            hist_row = obj.max_p_nts_lead+1;
            M = size(obj.MPRDarray,2);
            margcellX = cell(1,M);
            haspred = true;
            for ii=1:M
                X = zeros(size(scen,3),obj.npredvars(ii));
                Xcol = 1;
                for row=1:size(obj.MPRDarray,1)
                    MPRD = obj.MPRDarray{row,ii};
                    if ~isempty(MPRD)
                        [Xtmp, hp] = MPRD.make_predscen(...
                                hist_data,hist_row,scen,scenrowdate);
                        haspred = haspred&hp;
                        X(:,Xcol:Xcol+size(Xtmp,2)-1) = Xtmp;
                        Xcol = Xcol+size(Xtmp,2);
                    end
                end
                margcellX{1,ii} = X;
            end
        end
        function scenarios = make_scen_given_resp_and_scen(...
                    obj,cur_date,respscenprev,scenrowdate,respscencur)
            % load in the necessary historical data
            hist_data = obj.DH.get_data(...
                cur_date-obj.DTS(obj.max_real_nts_lead),...
                cur_date+obj.DTS(obj.max_real_nts_trail),false);
            hist_row = obj.max_real_nts_lead+1;
%             M = size(obj.CR2Rarray,2);
%             scenarios = zeros(M,obj.max_CR2R,size(respscencur,1));
            scenarios = zeros(size(respscencur,2),obj.max_CR2R,size(respscencur,1));
            for ii=1:size(respscencur,2)%M
                CR2R = obj.CR2Rarray{ii};
                nv = CR2R.get_num_vars();
                tmp = CR2R.make_variables(...
                    hist_data,hist_row,respscenprev,...
                    scenrowdate,respscencur(:,ii));
                scenarios(ii,1:nv,:) = tmp;
            end
        end
        function margcellXy = make_predresp(obj,hist_data)
            % use historical data to create a set of predictor and response
            % variables
            % inputs:
            %  - hist_data: NXT array of historical data
            % returns:
            %  - margcellXy: 1XM cell array
            %                each element ii is nXm_{ii} array, where the
            %                first m_{ii}-1 columns are the predictor
            %                variables and the last column is the response
            %                variable
            %
            % MPRRDarray is used to create the predictor and response
            % variables
            M = size(obj.MPRDarray,2);
            margcellXy = cell(1,M);
            lead  = obj.max_pr_nts_lead;
            trail = obj.max_pr_nts_trail;
            N = size(hist_data,1);
            for ii=1:M
                Xy = zeros(N-lead-trail,obj.npredvars(ii)+1);
                notnan = ones(size(Xy,1),1)>0;
                Xycol = 1;
                for row=1:size(obj.MPRDarray,1)
                    MPRD = obj.MPRDarray{row,ii};
                    if ~isempty(MPRD)
                        Xytmp = MPRD.make_predresp(hist_data,...
                                       obj.max_pr_nts_lead+1,...
                                       size(hist_data,1)-obj.max_pr_nts_trail);
                        notnan = notnan & ~isnan(Xytmp(:,end));
                        if Xycol>1
                            assert(all( abs(Xytmp(notnan,end)-Xy(notnan,end))<1e-8 ),...
                                   'response variables must be the same');
                        else
                            Xy(:,end) = Xytmp(:,end);
                        end
                        Xy(:,Xycol:Xycol+size(Xytmp,2)-2) = Xytmp(:,1:end-1);
                        Xycol = Xycol+size(Xytmp,2)-1;
                    end
                end
                margcellXy{1,ii} = Xy;
            end
        end
    end
    methods (Static)
        function test()
            GMD = GET_MARG_DATA();
            
            % make the marginal distributions -----------------------------
            % make 3 marginal distributions and combine up to two MPRDs
            MPRDarr = cell(2,3);
            % 1. lead time = 1, 
            MPRDarr{1,1} = MPRD_SIDES(1,0,0);
            % 2. lead time = 2
            MPRDarr{1,2} = MPRD_SIDES(2,0,0);
            MPRDarr{2,2} = MPRD_SIDES(2,0,1);
            % 3. lead time = 3
            MPRDarr{1,3} = MPRD_SIDES(3,1,1);
            % input
            GMD.input_MPRDarray(MPRDarr);
            
            % make the data handler ---------------------------------------
            DHinstance = DATA_HANDLER();
            % input
            GMD.input_DH(DHinstance);
            
            % set up some data for testing
            d1 = datenum(2013,12,30,0,0,0);
            dnts1 = 8;
            dnts2 = 5;
            N = 5;
%             % some bogus wind scenarios
%             nscen = 10;
%             scen = rand(N*(dnts1+dnts2+1),1,nscen);
            
            % constant window ---------------------------------------------
            % get the predictor and response variables 
            GMD.init_window_constant(d1,d1+datenum(0,0,0,N*(dnts1+dnts2+1)-1,0,0));
            [Xyc,uc] = GMD.get_predresp();
            for ii=1:size(uc,2)
                assert(uc{ii}==false)
            end
            [~,uc] = GMD.get_predresp();
            for ii=1:size(uc,2)
                assert(uc{ii}==true)
            end
            
            % sliding window ----------------------------------------------
            GMD.init_window_expand_slide('slide',dnts1,dnts2);
            row = 1;
            d = d1+datenum(0,0,0,dnts1,0,0);
            for ii=1:N
                [Xys,us] = GMD.get_predresp(d);
                for jj=1:size(Xyc,2)
                    tmp1 = Xyc{1,jj};
                    tmp2 = Xys{1,jj};
                    assert(size(tmp2,1)==dnts1+dnts2+1)
                    assert(all(all(...
                        abs(tmp1(row:row+size(tmp2,1)-1,:)-tmp2)<1e-8)))
                    for kk=1:size(us,2)
                        assert(us{kk}==false)
                    end
                end
                d = d + datenum(0,0,0,dnts1+dnts2+1,0,0);
                row = row+size(Xys{1,1},1);
            end
            
            % extending window --------------------------------------------
            GMD.init_window_expand_slide('expand',dnts1,dnts2);
            row = 1;
            d = d1+datenum(0,0,0,dnts1,0,0);
            for ii=1:N
                [Xye,ue] = GMD.get_predresp(d);
                for jj=1:size(Xyc,2)
                    tmp1 = Xyc{1,jj};
                    tmp2 = Xye{1,jj};
                    assert(size(tmp2,1)==dnts1+dnts2+1)
                    assert(all(all(...
                        abs(tmp1(row:row+size(tmp2,1)-1,:)-tmp2)<1e-8)))
                    for kk=1:size(ue,2)
                        assert(ue{kk}==(row>1))
                    end
                end
                d = d + datenum(0,0,0,dnts1+dnts2+1,0,0);
                row = row+size(Xye{1,1},1);
            end
            disp('passed test')
        end
    end
    
end