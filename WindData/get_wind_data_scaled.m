function [realization, forecast] = get_wind_data_scaled(year,month)
% same as get_wind_data(year,month), but realizations and forecasts are now
% scaled to [0,1]; see that function for inputs and outputs

% get unscaled data
[realization, forecast] = get_wind_data(year,month);

wind_capacity = check_wind_date_bounds(year,month); 

% get the start and end dates 
start_date = datenum(year,month,1,0,0,0);
if month<12
    end_date = datenum(year,month+1,1,0,0,0)-datenum(0,0,0,1,0,0);
else
    end_date = datenum(year+1,1,1,0,0,0)-datenum(0,0,0,1,0,0);
end

% check where data is NaN
rnan = isnan(realization);
fnan = isnan(forecast);
% make sure the maximum capacity is actually the maximum and then divide
c = get_capacity(start_date,end_date,wind_capacity);
realization = min(realization,c)./c;
for col=1:size(forecast,2)
    forecast(:,col) = min(forecast(:,col),c)./c;
end
% min(NaN,1) = 1 --> need to put back the NaNs
realization(rnan) = NaN;
forecast(fnan) = NaN;



end

function capacity = get_capacity(start_date,end_date,wind_capacity)
% get the wind capacity from start_date to end_date in hour increments
% start_date and end_date are datenums
d = start_date:datenum(0,0,0,1,0,0):end_date;
capacity = zeros(size(d,2),1);
wcdates = wind_capacity(:,1); % dates in wind_capacity
wcvals  = wind_capacity(:,2);  % capacity
for ii=1:size(wind_capacity,1)-1
    capacity(wcdates(ii)<=d & d<wcdates(ii+1)) = wcvals(ii);
end

end