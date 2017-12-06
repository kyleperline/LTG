classdef MPRD_F_SIDES < MAKE_PREDRESP_DATA
    % historical data is NXM array
    % first column is realization
    % columns 2:end are the forecasts
    % the response variable is the wind forecast
    % the predictor variables are previous time steps' forecast
    %
    % historical data = 
    %  [ r(1) f(1,2) f(1,3) f(1,4) f(1,5) ...
    %  [ r(2) f(2,3) f(2,4) f(2,5) f(2,6) ...
    %  [ r(3) f(3,4) f(3,5) f(3,6) f(3,7) ...
    %  [ r(4) f(4,5) f(4,6) f(4,7) f(4,8) ...
    % r(i)  = wind realization over time step (i-1) to i
    % f(i,j) = wind forecast created at the start of time step i that 
    %          predicts the wind realization rj
    % 
    % response variables at time step (row) i:
    %  f(i,j), j=horizon
    % predictor variables at time step (row) i:
    %  f(i-i2,j+j2), with 
    %     i2 = a1:a2
    %     j2 = min(maxhorizon-j, (a1:a2)+shift)
    %
    % e.g. 1 --------------
    %  horizon = 1
    %  maxhorizon = 3
    %  shift = 0
    %  a1 = 1
    %  a2 = 2
    %  i = 3
    %                               resp. var
    %  Xy = [ f(2,4)   , f(1,4)   , f(3,4)    ]
    %         pred. var, pred. var
    %
    % e.g. 2 ---------------
    %  horizon = 2
    %  maxhorizon = 3
    %  shift = -1
    %  a1 = 1
    %  a2 = 3
    %  i = 4
    %                                          resp. var
    %  Xy = [ f(3,5)   , f(2,5)   , f(1,4)   , f(4,6)    ]
    %         pred. var, pred. var, pred. var
    
    properties
        horizon
        maxhorizon
        shift
        prev1
        prev2
    end
    methods
        function d = get_description(obj)
            d = ['MPRD_F_SIDES',...
                    num2str(obj.horizon),'_',...
                    num2str(obj.maxhorizon),'_',...
                    num2str(obj.offset),'_',...
                    num2str(obj.prev1),'_',...
                    num2str(obj.prev2),'_'];
        end
        function obj = MPRD_F_SIDES(maxhorizon,horizon,shift,prev1,prev2,offset)
            obj = obj@MAKE_PREDRESP_DATA(offset);
            assert(prev1>=0)
            assert(prev2>=prev1)
            obj.maxhorizon = maxhorizon;
            obj.horizon    = horizon;
            obj.shift      = shift;
            obj.prev1      = prev1;
            obj.prev2      = prev2;
            % these three variables need to be defined from the parent
            % class
            obj.predresp_nts_lead  = prev2;
            obj.predresp_nts_trail = 0;
            obj.pred_nts_lead      = prev2;
        end
        function M = get_number_pred_vars(obj)
            M = obj.prev2-obj.prev1+1;
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
            Xy = zeros(row_end-row_start+1,obj.prev2-obj.prev1+2);
            % predictor variables
            for ii=1:obj.prev2-obj.prev1+1
                prev = obj.prev1+ii-1;
                col  = obj.horizon+1+obj.shift+prev;
                col  = max( min( col, 1+obj.maxhorizon) , 2);
                Xy(:,ii) = hist_data( (row_start-prev):(row_end-prev), col );
            end
            % reponse variable
            Xy(:,end) = hist_data(row_start:row_end,1+obj.horizon);
        end
        function [X, haspred] = make_predscen_fcn(obj,...
                hist_data,hist_row,scen_data,scen_row)
            % make the predictor variables given historical data and
            % generated scenarios
            % inputs:
            %  - hist_data: NXT matrix of historical wind data
            %               hist_data(:,1) is the wind realization
            %               hist_data(:,2:end) are the wind forecasts
            %               hist_data(ii,jj) predicts hist_data(ii+jj-1,1)
            %               for jj>1
            %  - hist_row  : positive integer
            %  - scen_data : nXHXS 3d array
            %                scen(ii,:,s) is the generated wind forecast at
            %                time step ii in scenario s
            %  - scen_row  : positive integer
            % returns:
            %  - X: SX(obj.prev2-obj.prev1+1) array
            %       X(s,:) are the predictor variables created as in 
            %       obj.make_predresp when hist_data(1:n,2:H) is set to
            %       scen(:,1:H,s)
            %  - haspred: boolean, true if all predictor variables could be
            %  created
            S = size(scen_data,3);
            X = zeros(S,obj.prev2-obj.prev1+1);
%             size(hist_data)
%             hist_data(1:11)
%             hist_row
%             size(scen_data)
%             scen_row
%             size(scen_data)
%             scen_row
%             disp('-----------------------------------------')
%             scen_row
%             scen_data
            for ii=1:obj.prev2-obj.prev1+1
                prev = obj.prev1+ii-1;
                if prev<=scen_row
                    % then the scenario is used instead of historical data
                    col  = obj.horizon+0+obj.shift+prev;
                    col  = max( min( col, 0+obj.maxhorizon) , 1);
                    Xii = reshape(scen_data(scen_row-prev+1,col,:),[],1);
                else
%                     hist_row
%                     prev
%                     hist_data(1:11)
                    % the scenarios do not go far enough back in time; need
                    % to use historical data
                    col  = obj.horizon+1+obj.shift+prev;
                    col  = max( min( col, 1+obj.maxhorizon) , 2);
                    Xii = hist_data(hist_row-prev,col)*ones(S,1);
                end
                X(:,ii) = Xii;
            end
            haspred = all(~isnan(X));
        end
    end
    methods (Static)
        function test()
            hist_data = [1.1 1.2 1.3 1.4 1.5;
                         2.1 2.2 2.3 2.4 2.5;
                         3.1 3.2 3.3 3.4 3.5;
                         4.1 4.2 4.3 4.4 4.5;
                         5.1 5.2 5.3 5.4 5.5];
            scen_data = zeros(size(hist_data,1),size(hist_data,2)-1,2);
            scen_data(:,:,1) = -hist_data(:,2:end);
            scen_data(:,:,2) = -10*hist_data(:,2:end);
            maxhorizon2=3;
            horizon2=1;
            shift2=-0;
            prev12=1;
            prev22=3;
            offset=0;
            ms = MPRD_F_SIDES(maxhorizon2,horizon2,shift2,prev12,prev22,offset);
            ms.make_predresp(hist_data,4,4)
            ms.make_predscen(hist_data,4,scen_data,4)
        end
            
    end
end











