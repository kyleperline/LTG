function [realization, forecast] = get_wind_data(year,month)
% get the actual wind realizations over all hours in specified year and
% month.  Time is in Pacific Time (PT).  There is no daylight savings.
% 
% Inputs:
%  - year: integer
%  - month: integer between 1 and 12 inclusive
%
% Outputs:
%  - realization: an NX1 matrix of wind realizations
%     N = 24*(number of days in month, year)
%     wind(ii) = wind realization over time (ii-1) to (ii) hours after
%     12 AM on the first day of the year, month
% - forecast: an NX72 matrix of wind forecasts
%     N = 24*(number of days in month, year)
%     forecast(ii,jj) = forecast created (ii) hours after 12 AM on first
%     date of the year, month over hour (ii+jj-1) to (ii+jj)
%     Some forecasts don't exist, in which case forecast(ii,jj) is NaN
% combining these two together, forecast(ii,jj) predicts realization(ii+jj)
%

%%%% UPDATE
% redid data
% [year month]
% first check data exists
if year==2013
    assert(month>=5 && month<=12,'in 2013, only have data for months 5 through 12')
elseif year==2014
    assert(month>=1 && month<=11,'in 2014, only have data for months 1 through 11')
else
    assert('year must be either 2013 or 2014')
end

% now get the month data
nhours = round(etime([year month+1 0 0 0 0],[year month 0 0 0 0])/3600);
offset = round(etime([year month 1 0 0 0],[2013 1 1 0 0 0])/3600);
D = load('realizations.mat');
delta = 7; % shift to Pacific Time Zone (yes, I know PT is 8 hours, not 7)
r = D.realizations(offset-7+delta:offset+nhours-8+delta);
if month~=12
    D = load(['forecasts_',num2str(year),'.mat']);
    if year==2014
        offset = offset-365*24;
    end
    % [size(D.(['forecasts_',num2str(year)]),1) offset nhours offset+nhours]
    f = D.forecasts(offset+1+delta:offset+nhours+delta,:);
else
    assert(year==2013)
    D = load('forecasts_2013.mat');
    f = D.forecasts(offset+1+delta:end,:);
    D = load('forecasts_2014.mat');
    f = [f;D.forecasts(1:delta,:)];
end

realization = r;
forecast = f;

% % scale data
% scale = 1;%4520*0.957;
% realization = r/scale;
% forecast = f/scale;

% some of the forecast information is bad ---
% instead of being NaNs, there are a couple of forecasts that are almost
% all 0s
% so, replace these with NaNs
iszero = sum(forecast(:,1:48),2)<eps;
forecast(iszero,:) = NaN;



% % first check that the requested data exists
% check_wind_date_bounds(year,month);
% 
% % update: -----------------------------------------------------------------
% % load in the new .mat files instead of loading excel files (which are
% % suuuuuper slow)
% % 'r' is from file 'BPA_historical_wind_data_',num2str(year),'.xls' and is
% % the realization in PT time over all hours in year 'year' and month
% % 'month'
% % 'f' is from files 
% % 'ForecastsMean_',num2str(year),'_',num2str(month),'.csv' and
% % 'ForecastsMean_',num2str(year2),'_',num2str(month2),'.csv', where
% % year2 and month2 are the next sequential year, month
% % 'f' is the mean BPA forecast in PT time over all hours in year, month
% 
% y = num2str(year);
% m = num2str(month);
% D = load(['wind_data_historical_and_mean_forecast_',y,'_',m,'.mat']);
% realization = D.r;
% % forecast = D.f;
% 
% % % update: realized the realizations and forecasts are off by two somehow
% forecast = D.f(3:end,:);
% if month==12
%     y = num2str(year+1);
%     m = '1';
% else
%     y = num2str(year);
%     m = num2str(month+1);
% end
% D = load(['wind_data_historical_and_mean_forecast_',y,'_',m,'.mat']);
% forecast = [forecast;D.f(1:2,:)];
% 
% % some of the forecast information is bad ---
% % instead of being NaNs, there are a couple of forecasts that are almost
% % all 0s
% % so, replace these with NaNs
% iszero = sum(forecast(:,1:48),2)<eps;
% forecast(iszero,:) = NaN;
% 
% 
% 
% 
% 
% 
% % y = num2str(year);
% % m = num2str(month);
% % load(['wind_data_historical_and_mean_forecast_',y,'_',m,'.mat'],'r','f');
% % realization = r;
% % forecast = f;
% 
% 
% 
% 
% % % old method: ------------------------------------------------------------
% % % get the start and end dates
% % start_date = datenum(year,month,1,0,0,0);
% % if month<12
% %     end_date = datenum(year,month+1,1,0,0,0)-datenum(0,0,0,1,0,0);
% % else
% %     end_date = datenum(year+1,1,1,0,0,0)-datenum(0,0,0,1,0,0);
% % end
% % 
% % % get the wind realization
% % % NOTE: this is in PT time
% % fname = strcat('BPA_historical_wind_data_',num2str(year),'.xls');
% % [realization,~,raw] = xlsread(fname);
% % dates = datenum(raw(1:end,1));
% % realization = realization(start_date<=dates & dates<=end_date);
% % 
% % % get the wind forecasts
% % % NOTE: this is in UTC time which is offset by 7 hours
% % TZ = 7; % time zone conversion
% % % so, we need to grab two months' data 
% % fname = strcat('ForecastsMean_',num2str(year),'_',num2str(month),'.csv');
% % forecast = xlsread(fname);
% % % get rid of the first 7 hours 
% % forecast = forecast(TZ+1:end,:);
% % % now get the first 7 hours of the next month
% % if month<12
% %     fname = strcat('ForecastsMean_',num2str(year),'_',num2str(month+1),'.csv');
% % else
% %     fname = strcat('ForecastsMean_',num2str(year+1),'_',num2str(1),'.csv');
% % end
% % f2 = xlsread(fname,strcat('1:',num2str(TZ)));
% % forecast = [forecast;f2];
% % % if forecast doesn't have the same number of rows as realization it's
% % % because the last line (or two or three...) of f2 are NaN and so didn't
% % % get copied. add these lines in
% % if size(forecast,1)<size(realization,1)
% %     forecast = [forecast;NaN(size(realization,1)-size(forecast,1),size(forecast,2))];
% % end

end


