% This shows examples of how to use Marginal Distributions with Kernel
% Density Estimation
% make the data
d = 1;
n = 100; % number of data points
minx = 0;
maxx = 1;
X = rand(n,d)*(maxx-minx)+minx;
y = sum(X,2).^2+0.05*randn(n,1);

miny = -1;
maxy = 2;
y = max(min(y,maxy),miny);
% make the 0 and 100 percentile functions
zerop = @(x) size(x,1)*miny; % y>=miny
onep  = @(x) size(x,1)*maxy; % y<=maxy

% make the marginal distribution
MD = MD_KDE();
MD.input_zero_one_percentiles(zerop,onep);
Kresp = KDEK_BETA(miny,maxy); % response kernel
MD.input_response_kernel(Kresp);
Kpred = {KDEK_BETA(minx,maxx)}; % predictor kernel
MD.input_predictor_kernels(Kpred);

% load the precomputed kernels
SK = KDE_SAVEDKERNEL();
SK = SK.load_beta();
MD.input_SavedKernel(SK);

% input the data, train
append_data = false; 
MD.input_data(X,y,append_data);
disp('-Training the marginal distribution (this will take about 25 seconds)')
MD.train();
training = MD.get_training();
disp('-We can retrain the marginal distribution using the previous training as an initial guess')
disp('(This will take about 8 seconds)')
MD.train(training);
disp('The optimized bandwidths will not be the same, but should be within about a factor of 10')
disp('-We can also input the bandwidths manually')
MD.input_pred_bandwidths(0.1*ones(1,d));
MD.input_resp_bandwidth(0.1);
disp('-Calculate the cumulative probability, p, of y conditioned on X')
c = MD.cdf(X,y);
disp('Now calculate the value yhat such that conditioned on X the cumulative probability is p')
y2 = MD.inverse_cdf(X,c);
disp(['we should have that y=yhat; maximum error = ',num2str(max(abs(y-y2)))])
disp('-Plot the cdf of y conditioned on X using a quantile plot')
MD.plot_quantiles_1d();
xlabel('predictor, X')
ylabel('response, cdf of y')






