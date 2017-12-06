classdef BPA_SCENARIOS
    % This file is used to generate the wind scenarios using data from
    % Bonneville Power Administration (BPA)
    methods (Static)
        function [GS,windscen,description] = ...
                make_forecast_scenarios()
            rng('shuffle')
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           
           retrain_every = 24*7; % retrain MD this # of time steps 
           myexpcov = 13; % Gaussian copula covariance 
           lambda0 = 0.01; % weight data points, 0< lambda0 <=1
                        % lambda0=1 -> all equally weighted
                        % lambda0=eps -> data points closer to current time
                        % step are more heavily weighted
           
           ntrainsampover2 = 10;%24*30; % half the number of training samples
           
           %---------------------------------------------------------------
           % T is the desired short-term scenario generation length
           % note that if historical data is missing then the short-term
           % scenarios may need to be longer 
           T = 72;
           % M is the number of marginal distributions
           % require M>=T
           M = T+0; % don't touch
           MPRDarr = cell(1,M);
           CR2Rarr = cell(1,M);
           do_error = false;  % hasn't been tested if set to true
           % whether to fill in historical data to use for training
           fill_data_train = false;
           % whether to fill in historical data to use for creating the
           % predictor variables used to create the wind scenarios
           fill_data_pred = true;
           % whether to fill in historical data in order to convert the
           % response scenarios into wind scenarios
           % (this should be true, unless you have a good reason to make it
           % false)
           fill_scen_data  = true;
           for ii=1:M
               % use forecasts
               offset = 0;
               MPRDarr{1,ii} = MPRD_SIDES(ii,0,0,do_error,offset);
               MPRDarr{1,ii}.set_fill_data_train(fill_data_train);
               MPRDarr{1,ii}.set_fill_data_pred(fill_data_pred);
               CR2Rarr{ii} = CR2R_TRAIL(ii,do_error,offset);
               CR2Rarr{ii}.set_fill_data(fill_scen_data);
           end
           GMDinstance.input_MPRDarray(MPRDarr);
           zopcell = GMDinstance.get_zero_one_percentiles();
           GMDinstance.input_CR2Rarray(CR2Rarr);
           
           % make the data handler ----------------------------------------
           DHinstance = DATA_HANDLER();
           % input
           GMDinstance.input_DH(DHinstance);
           % initialize the method of getting data ------------------------
           GMDinstance.init_window_expand_slide('slide',ntrainsampover2,ntrainsampover2);
           
           %---------------------------------------------------------------
           JDinstance = JD_GCECMHF(); % Gaus copula, exponential covariance
           JDinstance.input_cov_eps(myexpcov); % covariance parameter %%%%%%%%%%%%%%%%%%%%%%%
           
           
           MDarray = cell(1,M);
           SK = KDE_SAVEDKERNEL();
           SK = SK.load_beta();
           for ii=1:M
               assert(~do_error)
               md = MD_KDE();
               k1 = KDEK_BETA(0,1); % response kernel
               k2 = KDEK_BETA(0,1); % predictor kernel
               md.input_predictor_kernels({k2});
               md.input_response_kernel(k1);
               md.input_SavedKernel(SK);
               % retrain the marginal distribution every so often
               nts_train_every = retrain_every; % # time steps
               md.input_train_every(nts_train_every);
               % weight the data points 
               lambda = (lambda0)^(1/ntrainsampover2);
               md.input_data_weights_exp(lambda,ntrainsampover2);
               % specify the training objective function
               md.input_training_method('lik'); % either 'cdf' or 'lik'
               MDarray{ii} = md;
           end
           
           JDinstance.input_MDarray(MDarray);
           JDinstance.input_zero_one_percentiles(zopcell);
           
           
           %---------------------------------------------------------------
           GS = GEN_SCENARIOS();
           GS.input(JDinstance,GMDinstance);
           
           % NOTE: right now this only works with KDE, not QR
           % input the initial MD training
           % (see MD_KDE.get_training for details of MD_training)
           MD_training = struct('bw',[],...
                                'nts_since_last_trained',Inf);
           GS.input_initial_MD_training(MD_training);
           
           GS.input_T(T);
           d1 = datenum(2013,7,31,8,0,0);
           niters = 6;%14*3;
           nhours = 8;
           nscen = 10;
           tic
           windscen = GS.generateBPA(d1,niters,nhours,nscen);
           toc
           % plot
           DH = DATA_HANDLER();
           DH.init_constant_window();
           D = DH.get_data(d1-datenum(0,0,0,1,0,0),d1+datenum(0,0,0,niters*nhours,0,0));
           for ii=1:niters
               figure
               f = D((ii-1)*nhours+1,2:end);
               hold all
               plot(windscen(:,:,ii)')
               plot(f,'r','LineWidth',3)
           end
           
           % make a description 
           m = MDarray{1};
           description = struct(...
               'generation_method','NCM',...
               'ntrainsampover2',ntrainsampover2,...
               'fill_data_train',fill_data_train,...
               'fill_data_pred',fill_data_pred,...
               'fill_scen_data',fill_scen_data,...
               'M',M,...
               'T',T,...
               'MD',m.get_description(),...
               'JD',JDinstance.get_description(),...
               'd1',datestr(d1),...
               'nhours',nhours,...
               'niters',niters,...
               'nscen',nscen,...
               'lambda0',lambda0,...
               'retrain_every',retrain_every);
           for ii=1:size(MPRDarr,1)
               m = MPRDarr{ii,1};
               description.(['MPRDarr_',num2str(ii)]) = m.get_description();
           end
           
           % save
           version = 1;
           fname = 'BPA_wind_scenarios_v';
           while exist([fname,num2str(version),'.mat'],'file')
               version = version+1;
           end
           save([fname,num2str(version)],'windscen','description')
        end
    end
end