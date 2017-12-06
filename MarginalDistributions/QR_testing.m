clear all

qr = QUANTILE_REGRESSION('cq0X');
xmax = 2;
[x,y,zerop,onep] = qr.test_generate_samples_1d(xmax,1000);
qr = qr.input_data(x,y);
qr = qr.input_zero_one_percentiles(zerop,onep);
qr = qr.train();
qlist = [0,0.001,0.01,0.05, 0:0.1:0.9, 0.95,0.99,0.995,0.999,1];
xquant = (0:0.05:xmax)';
yquant = zeros(length(xquant),length(qlist));
for ii=1:length(qlist)
    yquant(:,ii) = qr.inverse_cdf(xquant,qlist(ii));
end
qr.plot_quantiles_1d(x,y,qlist,xquant,yquant)
[q,t] = qr.get_quantiles(xquant);