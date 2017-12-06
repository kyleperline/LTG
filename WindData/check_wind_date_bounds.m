function wind_capacity = check_wind_date_bounds(year,month)
% check that there is data for the requested year, month
% also return the maximum wind capacity
% This data is obtained from 
% https://transmission.bpa.gov/business/operations/wind/WIND_InstalledCapacity_PLOT.pdf
% NOTE: this is somewhat approximate since I read numbers off a chart

begin_time = datenum(2013,4,1,0,0,0);  % earliest month we have data
end_time   = datenum(2014,12,1,0,0,0); % last month we have data

cur_date = datenum(year,month,1,0,0,0);
msg = strcat('You requested wind data during a date for which there is no data. ', ...
             'You requested wind data for dates in the month of ',datestr(datenum(year,month,0,0,0,0)),...
             'There is only data between ',datestr(begin_time),' and ',datestr(end_time));
assert(begin_time<=cur_date && cur_date<=end_time,msg)

wind_capacity = [ datenum(2012,5,17,0,0,0)  , 4700 ; ...
                  datenum(2013,4,1,0,0,0)   , 4520*0.957 ; ...
                  datenum(2014,12,18,0,0,0) , 4900 ; ...
                  datenum(2016,7,1,0,0,0)   , 0      ];
              
end