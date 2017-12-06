classdef DATA_HANDLER < handle
    properties
        sew_data_r % some data stored in memory for the sliding window
        sew_data_f % some data stored in memory for the sliding window
        sew_start_date % dates for data in memory
        sew_end_date
        do_constant_window
        do_slide_extend_window
        data_source % string used to specify what data is returned
        custom_data
        custom_data_start
    end
    methods
       function obj = DATA_HANDLER()
           obj = obj.init_constant_window();
           obj.data_source = 'BPA';
       end
       function obj = input_data_source_string(obj,description)
           obj.data_source = description;
       end
       function obj = init_helper(obj)
           obj.do_constant_window = false;
           obj.do_slide_extend_window = false;
       end
       function obj = init_constant_window(obj)
           obj = obj.init_helper();
           obj.do_constant_window = true;
       end
       function obj = init_slide_extend_window(obj)
           obj = obj.init_helper();
           obj.do_slide_extend_window = true;
           obj.sew_start_date = NaN;
           obj.sew_end_date   = NaN;
           obj.sew_data_r     = [];
           obj.sew_data_f     = [];
       end
       function D = get_data(obj,start_date,end_date,clean_memory)
           % return inclusive data
           if nargin==3
               clean_memory = true;
           end
%            start_date = start_date - datenum(0,0,0,1,0,0);
           end_date = end_date + datenum(0,0,0,1,0,0);
           if obj.do_constant_window
               [r,f] = obj.get_seq_data(start_date,end_date);
           elseif obj.do_slide_extend_window
               [r,f] = obj.get_se_window_data(...
                                  start_date,end_date,clean_memory);
           else
               error('nope');
           end
           D = [r,f];
       end
       function [r,f] = get_se_window_data(obj,start_date,end_date,clean_memory)
           % get wind data
           %
           % UPDATE: This whole function isn't really needed anymore - it
           % turns out the whole issue with loading data was loading data
           % from Excel - loading from .mat files doesn't take forever
           % This function by itself works, but it hasn't been incorporated 
           % into the rest of the main code
           % END UPDATE
           %
           % this method is used to provide a balance between loading all
           % data into memory at once (which can be a lot) with the time
           % required to load data from a file
           %
           % a contiguous set of data is stored and updated in memory
           % corresponding to the data between dates SD and ED
           % 
           % suppose the the method is called with
           % >> get_se_window_data(sd,ed,clean_memory)
           % If sd<SD then an error is raised
           % If sd>=SD then the method first checks if SD<=sd and ed<=ED
           %   If so, then all data is already loaded into memory
           % Else more data is loaded and ED is updated so that ed<=ED
           % If clean_memory=true (default), then some of the stored data
           %   is removed and SD is updated so that SD=sd
           % Else clean_memory=false and no stored data is removed
           % 
           % Some usage examples:
           % >> get_se_window_data(sd_1,ed_1,true)
           % >> get_se_window_data(sd_2,ed_2,false) % error iff sd_2<sd_1
           % >> get_se_window_data(sd_3,ed_3,true)  % error iff sd_3<sd_1
           % >> get_se_window_data(sd_4,ed_4,true)  % error iff sd_4<sd_3
           a=asdf;
           % first check that these are acceptable dates
           assert(start_date<=end_date,...
                  'end_date must be later than start_date')
              
           if nargin<4
               clean_memory = true;
           end
           
           if ~isnan(obj.sew_start_date)
               assert(obj.sew_start_date<=start_date,...
                   'you did not call get_se_window_date sequentially')
           end
           % first check to see if all the requested data is already in
           % memory
           load_more = false;
           y1 = year(start_date);
           m1 = month(start_date);
           y2 = year(end_date);
           m2 = month(end_date);
           if isnan(obj.sew_start_date)
               obj.sew_start_date = datenum(y1,m1,1,0,0,0);
               load_more = true;
           else
               if end_date>obj.sew_end_date
                   y1 = year(obj.sew_end_date);
                   m1 = month(obj.sew_end_date);
                   [y1,m1] = obj.get_next_ym(y1,m1);
                   load_more = true;
               end
           end
           if load_more
               % we need data starting from year y1, month m1
               [r,f] = obj.get_seq_month_data(y1,m1,y2,m2);
               % update the data 
               obj.sew_data_r = [obj.sew_data_r; r];
               obj.sew_data_f = [obj.sew_data_f; f];
               [y3,m3] = obj.get_next_ym(year(end_date),month(end_date));
               obj.sew_end_date = datenum(y3,m3,1,0,0,0)-...
                                              datenum(0,0,0,1,0,0);
           end
           
           % delete the stored data we no longer need and update
           % sew_start_date
           if clean_memory
               nremove = floor(etime(datevec(start_date),...
                                     datevec(obj.sew_start_date))/3600);
               obj.sew_data_r = obj.sew_data_r(nremove+1:end,1);
               obj.sew_data_f = obj.sew_data_f(nremove+1:end,:);
               obj.sew_start_date = start_date;
           end
           % and finally get r, f
           n = floor(etime(datevec(end_date),datevec(start_date))/3600);
           r = obj.sew_data_r(1:n,1);
           f = obj.sew_data_f(1:n,:);
       end

    end
   methods (Static)
       function [y,m] = get_next_ym(y,m)
           if m<12
               m = m+1;
           else
               y = y+1;
               m = 1;
           end
       end
   end
   methods
       function [r,f] = get_seq_data_inclusive(obj,start_date,end_date)
           % returns data for dates start_date to end_date
           [r,f] = obj.get_seq_data(start_date,end_date+datenum(0,0,0,1,0,0));
       end
       function [r,f] = get_seq_data(obj,start_date,end_date)
           % returns data for dates (start_date+1 hour) to end_date
           % (in retrospect, I'm not sure why this skips the first date)
           [r,f] = obj.get_seq_month_data(...
                             year(start_date),month(start_date),...
                             year(end_date),  month(end_date));
            % get the number of hours missing at the start of the fist
            % month
            nstart = floor(etime(datevec(start_date),...
              datevec(datenum(year(start_date),month(start_date),1,0,0,0)))/3600);
            % get the number of hours missing at the end of the last month
            y2 = year(end_date);
            m2 = month(end_date);
            [y3,m3] = obj.get_next_ym(y2,m2);
            nend = floor(etime(...
                datevec(datenum(y3,m3,1,0,0,0)),datevec(end_date))/3600);
            r = r(nstart+1:end-nend,1);
            f = f(nstart+1:end-nend,:);
            % add in custom data if there is any
            if ~isempty(obj.custom_data)
                start_row = floor(etime(datevec(start_date),datevec(obj.custom_data_start))/3600);
                end_row   = floor(etime(datevec(end_date),  datevec(obj.custom_data_start))/3600);
                [m,n] = size(obj.custom_data);
                nrows = end_row-start_row;
                mydata = NaN(nrows,n);
                if end_row<1 || start_row>m
                    data_start_row = 1;
                    data_end_row   = 0;
                    start_row      = 1;
                    end_row        = 0;
                else
                    if start_row<1
                        data_start_row = min(nrows,1-start_row);
                        start_row      = 1;
                    else
                        data_start_row = 1;
                        start_row      = start_row+1;
                    end
                    if end_row>m
                        data_end_row = nrows-(end_row-m);
                        end_row      = m;
                    else
                        data_end_row = nrows;
                    end
                end
                mydata(data_start_row:data_end_row,:) = obj.custom_data(start_row:end_row,:);
                f = [f mydata];
            end
            
       end
       function [r,f] = get_scaled_month_data_base(obj,y1,m1)
           bta = false;
           if strcmp(obj.data_source,'BPA')
               [r,f] = get_wind_data_scaled(y1,m1);
%            elseif strcmp(obj.data_source,'tmp')
%                [~,f] = get_wind_data_scaled(y1,m1);
%                D = load('tmp.mat');
%                bta = true;
           elseif strcmp(obj.data_source,'bt1')
               D = load('betadist1_vs6_pa96_pev0_peb0_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt2')
               D = load('betadist2_vs6_pa96_pev0.03_peb0_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt3')
               D = load('betadist3_vs6_pa96_pev0_peb0.03_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt4')
               D = load('betadist4_vs6_pa96_pev0.03_peb0_pec1.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt1norm')
               D = load('normdist1_vs1.4_pa120_pev0_peb0_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt2norm')
               D = load('normdist2_vs1.4_pa120_pev0.03_peb0_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt3norm')
               D = load('normdist3_vs1.4_pa120_pev0_peb0.03_pec0.mat');
               bta = true;
           elseif strcmp(obj.data_source,'bt4norm')
               D = load('normdist4_vs1.4_pa120_pev0.03_peb0_pec1.mat');
               bta = true;
           else
               error(['unrecognized data_source: ',obj.data_source])
           end
           if bta
               r = D.y;
               f = D.x;
               % index 1 is [2013 4 1 0 0 0]
               ind1 = round(etime([y1 m1 1 0 0 0],[2013 4 1 0 0 0])/3600)+1;
               ind2 = round(etime([y1 m1+1 1 0 0 0],[2013 4 1 0 0 0])/3600);
               r = r(ind1:ind2);
               f = f(ind1:ind2);
           end
       end
       function obj = input_custom_data(obj,data,start)
           % input some custom data
           % inputs:
           %  - data : nXm array
           %  - start: datenum, data(ii,:) corresponds to data at date
           %           start+datenum(0,0,0,ii-1,0,0)
           obj.custom_data = data;
           obj.custom_data_start = start;
       end
       function [r,f] = get_seq_month_data(obj,y1,m1,y2,m2)
           assert(etime(datevec(datenum(y2,m2,0,0,0,0)),...
                        datevec(datenum(y1,m1,0,0,0,0)))>=0)
%            [r1,f1] = get_wind_data_scaled(y1,m1); % replaced with below
           [r1,f1] = obj.get_scaled_month_data_base(y1,m1);
           [y3,m3] = obj.get_next_ym(y2,m2);
           N = floor(etime(datevec(datenum(y3,m3,1,0,0,0)),...
                           datevec(datenum(y1,m1,1,0,0,0)))/3600);
           r = zeros(N,1);
           f = zeros(N,size(f1,2));
           r(1:size(r1,1),1) = r1;
           f(1:size(f1,1),:) = f1;
           row = size(f1,1)+1;
           ct = 0;
           while ~(y1==y2 && m1==m2)
               [y1,m1] = obj.get_next_ym(y1,m1);
%                [r1,f1] = get_wind_data_scaled(y1,m1); % replaced 
               [r1,f1] = obj.get_scaled_month_data_base(y1,m1);
               r(row:row+size(r1,1)-1,1) = r1;
               f(row:row+size(f1,1)-1,:) = f1;
               row = row+size(f1,1);
               ct = ct+1;
               if ct==10000
                   warning('DH:idk','you may be stuck in an infinite loop')
               end
           end
           assert(row==N+1,'I messed this up.')
       end
   end
   methods (Static)
       function test_timing()
           % do some timing tests
           DH = DATA_HANDLER();
           DH.init_constant_window();
           d = 24;
           delta = 12;
           N = 10;
           d1 = datenum(2013,6,28,0,0,0);
           data = cell(1,N);
           % constant window method
           tic
           for ii=1:N
               D = DH.get_data(d1+datenum(0,0,0,ii*delta  ,0,0),...
                           d1+datenum(0,0,0,ii*delta+d,0,0));
               data{ii} = D;
               assert(size(D,1)==d+1)
           end
           t = toc;
           disp(['constant window, average time: ',num2str(t/N)]);
           % slide/extend window method
           DH.init_slide_extend_window();
           tic
           for ii=1:N
               D = DH.get_data(d1+datenum(0,0,0,ii*delta  ,0,0),...
                           d1+datenum(0,0,0,ii*delta+d,0,0));
               assert(size(D,1)==d+1)
               % compare this to the constant window method
               D(isnan(D)) = 0;
               D2 = data{ii};
               D2(isnan(D2)) = 0;
               assert(all(all(abs(D-D2)<1e-8)))
           end
           t = toc;
           disp(['slide/extend window, average time: ',num2str(t/N)]);
           disp('passed tests')
       end
   end
    
end