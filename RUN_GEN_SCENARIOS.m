classdef RUN_GEN_SCENARIOS
    methods (Static)
        function results_struct = analyze_scenarios(windscen,obs)
            % analyze windscen using ERROR_STATS
            % windscen is output from GENERATE_SCENARIOS.generate, and is
            % an array of size TX1XN, where N=number of scenarios
            % obs is TX1 array of the historical wind realizations over the
            % corresponding time, so n = size(windscen,1)
            % returns:
            %  results_struct.Brier_up: 3X2 array of ramp up brier scores
            %     (see function get_brier_kappa_xi() in ERROR_STATS.m for
            %     kappa, xi)
            %  results_struct.Brier_down: 3X2 array of ramp down brier
            %     scores
            %  results_struct.MST_results_cell: 1Xlength(deltatlist) cell
            %     array
            %     deltatlist is a list of positive integers defined below
            %     set ri = results_struct.MST_results_cell{1,i}, i=1,...
            %         dt = deltatlist{i}
            %         nscen = 50 (this is hard coded)
            %     then the MST ranks are first calculated for each set 
            %       { windscen(dt*(t-1)+1:dt*t, 1, nscen*(n-1)+1:nscen*n)
            %       { obs(dt*(t-1)+1:dt*t,1)
            %       for t=1,...,floor(T/dt) and n=1,...,floor(N/nscen)
            %     ri is a nsplitXfloor(N/nscen) cell array
            %       to get rank counts of all floor(T/dt)*floor(N/nscen)
            %       ranks do:
            %       >> rank_sum = 0;
            %       >> for row=1:size(ri,1)
            %       >>   for col=1:size(ri,2)
            %       >>     rank_sum = ri{row,col}.MST_rank_count;
            %       >> end, end
            %       rank_sum is a 1X(nscen+1) array
            w = reshape(windscen,size(windscen,1),size(windscen,3));
            assert(size(w,1)==size(obs,1));
            [T,N] = size(w);
            % w is TXD matrix = T time steps X N scenarios
            % partition w into rectangles, calculate results on each of
            % these sets, and then combine all the results together
            
            % Brier Scores ------------------------------------------------
            [Brier_down, Brier_up] = ...
                ERROR_STATS.compare_gen2obs_single_Brier(w',obs');
            
            % MST ---------------------------------------------------------
            % partition:
            % number of rows (time steps) in partition set
            deltatlist = 24;% unique(min([1,3,6,12,24,24*4 ],T));
            % number of scenarios in partition sets
            nscen = min(50,N);
            MST_results_cell = cell(1,length(deltatlist));
            for ii=1:length(deltatlist)
                dt = deltatlist(ii);
                results = cell(floor(T/dt),floor(N/nscen)); 
                for row=1:floor(T/dt)
                    r1 = (row-1)*dt+1;
                    r2 = row*dt;
                    % get Mahalonobis transformation
                    mydata = [w(r1:r2,:),obs(r1:r2)]';
                    mycov  = nancov(mydata);
                    [V,D]  = eig(mycov);
                    S2 = V*(eye(size(D,1)).*(1./sqrt(diag(D))*ones(1,size(D,1))))*V;
                    if sum(sum( abs( (S2-real(S2)).^2 ) ))>1e-10
                        % this happens if size(mydata,1)<dt
                        S2 = eye(size(S2,1));
                    end
                    
                    for col=1:floor(N/nscen) 
                        c1 = (col-1)*nscen+1;
                        c2 = col*nscen;
                        results{row,col} = ERROR_STATS.compare_gen2obs_single_MST(...
                            w(r1:r2,c1:c2)'*S2,obs(r1:r2)'*S2); 
                    end
                end

                % combine results into a cell array, put this in results_cell
                % combine in the following manner:
                nsplit = min(ceil(24/dt)+1,size(results,1));
                results_combined = cell(nsplit,size(results,2));
                size(results_combined)
                for row = 1:nsplit
                    for col=1:size(results,2)
                        
                        myrows = row:nsplit:size(results,1);
                        mycell = cell(length(myrows),1);
                        for kk=1:length(mycell)
                            mycell{kk} = results{myrows(kk),col};
                        end
                        rav = ERROR_STATS.combine_single_results_MST(mycell);
                        rav.nsplit = nsplit;
                        rav.individual_results_n_time_steps = dt;
                        rav.individual_results_n_scenarios  = nscen;
                        results_combined{row,col} = rav;
                    end
                end
                MST_results_cell{1,ii} = results_combined;
            end
            results_struct = struct('MST_results_cell',{MST_results_cell},...
                                    'MST_results',{results},...
                                    'Brier_down',Brier_down,...
                                    'Brier_up',Brier_up);
        end
        function save_results(fname)
            % call this function to save the important variables
            global windscen results_struct description
            % check if file exists
            if exist(fname,'file')==2
                error('that file name is already in use')
            end
            save(fname,'windscen','results_struct','description')
        end
        function [GS,respscen,windscen,results_struct,success,description] = ...
                LTG()
           rng('shuffle')
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           
           retrain_every = 24*7; % retrain MD this # of time steps 
           myexpcov = 9; % Gaussian copula covariance 
           lambda0 = 1; % weight data points, 0< lambda0 <=1
                        % lambda0=1 -> all equally weighted
                        % lambda0=eps -> data points closer to current time
                        % step are more heavily weighted
           
           time_horizon = 24*366;
           ntrainsampover2 = 24*30; % half the number of training samples
           %%% predictor kernels:
           % if iterate_predlags_single==true
           %   then at time step t the predictor kernel is
           %   S^{t-pl}_pl, where pl = predlag(mod(t,length(predlag))+1)
           % else
           %   then at time step t there are length(predlag) predictor
           %   predictor kernels, and they are
           %   { S^{t-predlag(1)}_{predlag(1)}, ... ,
           %       S^{t-predlag(end)}_{predlag(end)} }
           % end
           % predictor variables are S^{t-pl}_pl for pl=predlag
           iterate_predlags_single = false;
           predlag = 1; % a list, can be empty
           predlag = sort(predlag); % don't touch
           % can also (or instead) use the first PCA_ncomp PCA variables as
           % the predictor variables
           % do PCA on first PCA_horizon components of the forecast
           % i.e. the PCA_horizon hour-ahead forecast
           do_PCA = false;
           PCA_horizon = 24; 
           PCA_ncomps = 2;
           
           % --------------------------------------------------------------
           % make the marginal distributions and real-world conversions ---
           % make M marginal distributions, at each time step generate 
           % scenarios for 1<=T<M time steps
           M = time_horizon+2; % don't touch
           T = M-1; % don't touch
            
           do_error = false; % hasn't been tested if set to true
           % whether to fill in historical data to use for training
           % NOTE: if do_PCA==true, then fill_data_train should be true;
           %  otherwise, there will probably not be enough training samples
           fill_data_train = do_PCA;
           
           % whether to fill in historical data in order to create the
           % predictor variables
           % NOTE: historical data is always filled in for PCA;
           % fill_data_pred only controls predlag predictor variables
           fill_data_pred = false; % for predlag
           
           % whether to fill in historical data in order to convert the
           % response scenarios into wind scenarios
           % (this should be true, unless you have a good reason to make it
           % false)
           fill_scen_data  = true;
           
           % don't touch any of this:
           % --------------------------------------------------------------
           % don't touch:
           if ~isempty(predlag)
               predlagoffset = 72;
           else
               predlagoffset = 0;
           end
           d1 = datenum(2013,7,31,0,0,0);
           d2 = datenum(2013,7,31,time_horizon,0,0);
           % make the data handler 
           DHinstance = DATA_HANDLER(); % don't touch
           [~,f] = DHinstance.get_seq_data_inclusive(...
               d1+datenum(0,0,0,-predlagoffset,0,0),...
               d1+datenum(0,0,0,M,0,0));
           fnan = isnan(f);
           % end don't touch
           % --------------------------------------------------------------
           
           if ~isempty(predlag)
               if iterate_predlags_single
                   totalpredvars = 1;
               else
                   totalpredvars = length(predlag);
               end
           else
               totalpredvars = 0;
           end
           MPRDarr = cell(totalpredvars+do_PCA,M);
           CR2Rarr = cell(totalpredvars+do_PCA,M);
           if do_PCA
               totalpredvars = totalpredvars + PCA_ncomps;
           end
           MDtypelist = zeros(1,M);
           for ii=1:M
               offset = T-M-1+ii;
               % do predlag 
               if ~isempty(predlag) && iterate_predlags_single
                   % only a single predictor variable
                   curplag = mod(ii,length(predlag))+1;
                   curplag = predlag(curplag);
                   if ~fill_data_pred && fnan(ii-1+predlagoffset-curplag,curplag)
                       % then missing historical forecasts are not filled
                       % in using persistence
                       % this means we need to check that the predictor
                       % variable exists
                       % if it doesn't, then we need to change the
                       % predictor variable
                       while fnan(ii-1+predlagoffset-curplag,curplag)
                           curplag = curplag + 1;
                       end
                   end
                   MPRDarr{1,ii} = MPRD_SIDES(curplag,0,0,do_error,offset+1-curplag);
                   MPRDarr{1,ii}.set_fill_data_train(fill_data_train);
                   MPRDarr{1,ii}.set_fill_data_pred(fill_data_pred);
                   CR2Rarr{1,ii} = CR2R_TRAIL(curplag,do_error,offset+1-curplag);
                   CR2Rarr{1,ii}.set_fill_data(fill_scen_data);
                   currow = 2;
                   MDtypelist(ii) = curplag;
               elseif ~isempty(predlag) && ~iterate_predlags_single
                   npredlag = length(predlag);
                   if fill_data_pred
                       predlagtmp = predlag;
                       for pv=1:npredlag
                           curplag = predlag(pv);
                           MPRDarr{pv,ii} = MPRD_SIDES(curplag,0,0,do_error,offset+1-curplag);
                           MPRDarr{pv,ii}.set_fill_data_train(fill_data_train);
                           MPRDarr{pv,ii}.set_fill_data_pred(fill_data_pred);
                           CR2Rarr{pv,ii} = CR2R_TRAIL(curplag,do_error,offset+1-curplag);
                           CR2Rarr{pv,ii}.set_fill_data(fill_scen_data);
                       end
                   else
                       predlagtmp = zeros(1,npredlag);
                       curplag = predlag(1)-1;
                       for pv=1:npredlag
                           curplag = max(curplag+1,predlag(pv));
                           while fnan(ii-1+predlagoffset-curplag,curplag)
                               curplag = curplag + 1;
                           end
                           predlagtmp(pv) = curplag;
                           MPRDarr{pv,ii} = MPRD_SIDES(curplag,0,0,do_error,offset+1-curplag);
                           MPRDarr{pv,ii}.set_fill_data_train(fill_data_train);
                           MPRDarr{pv,ii}.set_fill_data_pred(fill_data_pred);
                           CR2Rarr{pv,ii} = CR2R_TRAIL(curplag,do_error,offset+1-curplag);
                           CR2Rarr{pv,ii}.set_fill_data(fill_scen_data);
                       end
                   end
                   currow = npredlag+1;
                   MDtypelist(ii) = sum( predlagtmp.*( (72*ones(1,npredlag)).^(0:(npredlag-1)) ) );
               else
                   currow = 1;
               end
               % do PCA
               if do_PCA
                   MPRDarr{currow,ii} = MPRD_PCA(PCA_horizon,PCA_ncomps,do_error,offset);
                   MPRDarr{currow,ii}.set_fill_data_train(fill_data_train);
                   MPRDarr{currow,ii}.set_fill_data_pred(true);
                   CR2Rarr{currow,ii} = CR2R_TRAIL(1,do_error,offset);
                   CR2Rarr{currow,ii}.set_fill_data(fill_scen_data);
               end
           end
           
           GMDinstance.input_MPRDarray(MPRDarr);
           zopcell = GMDinstance.get_zero_one_percentiles();
           GMDinstance.input_CR2Rarray(CR2Rarr);
           % input Data Handler instance
           GMDinstance.input_DH(DHinstance); % don't touch
           
           % initialize the method of getting data ------------------------
           GMDinstance.init_window_expand_slide('slide',ntrainsampover2,ntrainsampover2);
           
           % make the joint distribution ----------------------------------
           % Gaussian copula, time adaptive
%            JDinstance = JD_GCTAEF(); 
%            JDinstance.input_lambda(0.99);
           % Gaussian copula, exponential covariance
           JDinstance = JD_GCECMHF(); 
           JDinstance.input_cov_eps(myexpcov); % covariance parameter 
           % Gaussian copula, empricial covariance
%            JDinstance = JD_GCEMP();
           
           MDarray = cell(1,M);
%            % QR:
%            for ii=1:M
%                md = MD_QUANTILE_REGRESSION('c');
%                MDarray{ii} = md;
%            end
           % KDE:
           SK = KDE_SAVEDKERNEL();
           SK = SK.load_beta();
           for ii=1:M
               assert(~do_error)
               md = MD_KDE();
               k1 = KDEK_BETA(0,1); % response kernel
               mypredkernels = cell(1,totalpredvars);
               for qq = 1:totalpredvars
                   mypredkernels{qq} = KDEK_BETA(0,1);
               end
               md.input_predictor_kernels(mypredkernels);
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
%            MD_training = {[], Inf}; % (see MD_KDE.get_training for details)
           GS.input_initial_MD_training(MD_training);
           
           GS.input_T(T);
           nscen = 100;
           tic
           [respscen,windscen,success] = GS.generateLTG(d1,d2,nscen,MDtypelist);
           % this method starts generating scenarios at M-T time steps
           % earlier. the scenarios at these initial time steps should be
           % exactly equal to the wind realization, so this is a good check
%            GS.plot_scen(respscen,d1-datenum(0,0,0,M-T,0,0),d2,windscen,do_error)
           
           % respscen and windscen have results from time steps that are
           % not between d1, d2. get rid of excess
           r = M-T+1;
           nrows = round(etime(datevec(d2),datevec(d1))/3600)+1;
           windscen = windscen(r:r+nrows-1,:,:);
           respscen = respscen(r:r+nrows-1,:,:);
           % plot this to double check we did things right
           GS.plot_scen(respscen,d1,d2,windscen,do_error)
           toc
           tic
           
           % now do error analysis
           if success
               % need to get the wind observation from dates d1 to d2
               obs = DHinstance.get_seq_data_inclusive(d1,d2);
               % include a burn-in period - the first couple of scenarios
               % are all conditioned on the same historical previous data
               burn_in = 24; % number of time steps
               results_struct = RUN_GEN_SCENARIOS.analyze_scenarios(...
                   windscen(burn_in+1:end,:,:),obs(burn_in+1:end));
           end
           toc
           
           % make a description 
           m = MDarray{1};
           description = struct(...
               'generation_method','LTG',...
               'time_horizon',time_horizon,...
               'ntrainsampover2',ntrainsampover2,...
               'fill_data_train',fill_data_train,...
               'fill_data_pred',fill_data_pred,...
               'fill_scen_data',fill_scen_data,...
               'M',M,...
               'T',T,...
               'MD',m.get_description(),...
               'JD',JDinstance.get_description(),...
               'd1',datestr(d1),...
               'd2',datestr(d2),...
               'nscen',nscen,...
               'predlag',predlag,...
               'iterate_predlags_single',iterate_predlags_single,...
               'do_PCA',do_PCA,...
               'PCA_ncomps',PCA_ncomps,...
               'pPCA_horizon',PCA_horizon,...
               'lambda0',lambda0,...
               'retrain_every',retrain_every);
           for ii=1:size(MPRDarr,1)
               m = MPRDarr{ii,1};
               description.(['MPRDarr_',num2str(ii)]) = m.get_description();
           end
           
           
        end
        function [GS,respscen,windscen,results_struct,success,description] = ...
                NCM()
           rng('shuffle')
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           
           retrain_every = 24*7; % retrain MD this # of time steps 
           myexpcov = 9; % Gaussian copula covariance 
           lambda0 = 1; % weight data points, 0< lambda0 <=1
                        % lambda0=1 -> all equally weighted
                        % lambda0=eps -> data points closer to current time
                        % step are more heavily weighted
           
           time_horizon = 48;%24*366;
           ntrainsampover2 = 100;%24*30; % half the number of training samples
           
           %---------------------------------------------------------------
           % T is the desired short-term scenario generation length
           % note that if historical data is missing then the short-term
           % scenarios may need to be longer 
           T = 4;
           % M is the number of marginal distributions
           % require M>=T
           M = T+9; % don't touch
           MPRDarr = cell(1,M);
           CR2Rarr = cell(1,M);
           do_error = false;  % hasn't been tested if set to true
           % whether to fill in historical data to use for training
           fill_data_train = false;
           % whether to fill in historical data to use for creating the
           % predictor variables used to create the wind scenarios
           fill_data_pred = false;
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
%            JDinstance = JD_GCTAEF();
%            JDinstance.input_lambda(0.99);
           JDinstance = JD_GCECMHF(); % Gaus copula, exponential covariance
           JDinstance.input_cov_eps(myexpcov); % covariance parameter %%%%%%%%%%%%%%%%%%%%%%%
           
           
           MDarray = cell(1,M);
%            % QR:
%            for ii=1:M
%                md = MD_QUANTILE_REGRESSION('c');
%                MDarray{ii} = md;
%            end
           % KDE:
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
%            MD_training = {[], Inf}; % (see MD_KDE.get_training for details)
           GS.input_initial_MD_training(MD_training);
           
           GS.input_T(T);
           d1 = datenum(2013,7,31,0,0,0);
           d2 = datenum(2013,7,31,time_horizon,0,0);
           nscen = 100;
           tic
           [respscen,windscen,success] = GS.generateNCM(d1,d2,nscen);
           % respscen and windscen have results from time steps that are
           % not between d1, d2. get rid of excess
           r = 1;
           nrows = round(etime(datevec(d2),datevec(d1))/3600)+1;
           windscen = windscen(r:r+nrows-1,:,:);
           respscen = respscen(r:r+nrows-1,:,:);
           % plot this to double check we did things right
           GS.plot_scen(respscen,d1,d2,windscen,do_error)
           toc
           tic
           
           % now do error analysis
           if success
               % need to get the wind observation from dates d1 to d2
               DH = DATA_HANDLER();
               obs = DH.get_seq_data_inclusive(d1,d2);
               % include a burn-in period - the first couple of scenarios
               % are all conditioned on the same historical previous data
               burn_in = 21; % number of time steps
               results_struct = RUN_GEN_SCENARIOS.analyze_scenarios(...
                   windscen(burn_in+1:end,:,:),obs(burn_in+1:end));
           end
           
           % make a description 
           m = MDarray{1};
           description = struct(...
               'generation_method','NCM',...
               'time_horizon',time_horizon,...
               'ntrainsampover2',ntrainsampover2,...
               'fill_data_train',fill_data_train,...
               'fill_data_pred',fill_data_pred,...
               'fill_scen_data',fill_scen_data,...
               'M',M,...
               'T',T,...
               'MD',m.get_description(),...
               'JD',JDinstance.get_description(),...
               'd1',datestr(d1),...
               'd2',datestr(d2),...
               'nscen',nscen,...
               'lambda0',lambda0,...
               'retrain_every',retrain_every);
           for ii=1:size(MPRDarr,1)
               m = MPRDarr{ii,1};
               description.(['MPRDarr_',num2str(ii)]) = m.get_description();
           end

        end
        function [GS,respscen,windscen,results_struct,success,description] = ...
                beta_analysis_generation(data_source,horizon)
           rng('shuffle')
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           
           time_horizon = 24*366;%24*365;
           ntrainsampover2 = 24*30; % half the number of training samples
           % predictor kernels:
           %   predvars=1 -> S^{t-1}_1
           predvars = 1; % only option
           
           % --------------------------------------------------------------
           % make the marginal distributions and real-world conversions ---
           % make M marginal distributions, at each time step generate 
           % scenarios for 1<=T<M time steps
           M = time_horizon+2;
           T = M-1;
           
            % predictor variables = (S^{t-1}_1,S^{t-2}_2) -----------------
           
           do_error = false; % hasn't been tested if set to true
           % whether to fill in historical data to use for training
           fill_data_train = false;
           % whether to fill in historical data in order to convert the
           % response scenarios into wind scenarios
           % (this should be true, unless you have a good reason to make it
           % false)
           fill_scen_data  = true;
           if predvars==1
               MPRDarr = cell(1,M);
               CR2Rarr = cell(1,M);
           end
           for ii=1:M
               offset = T-M-1+ii;
               if predvars==1
                   % S^{t-1}_1
                   MPRDarr{1,ii} = MPRD_SIDES(horizon,0,0,do_error,offset+1-horizon);
                   MPRDarr{1,ii}.set_fill_data_train(fill_data_train);
                   CR2Rarr{1,ii} = CR2R_TRAIL(horizon,do_error,offset+1-horizon);
                   CR2Rarr{1,ii}.set_fill_data(fill_scen_data);
               end
           end
           
           
           GMDinstance.input_MPRDarray(MPRDarr);
           zopcell = GMDinstance.get_zero_one_percentiles();
           GMDinstance.input_CR2Rarray(CR2Rarr);
           
           % make the data handler ----------------------------------------
           DHinstance = DATA_HANDLER();
           % input the data source
           DHinstance.input_data_source_string(data_source);
           % input
           GMDinstance.input_DH(DHinstance);
           
           % initialize the method of getting data ------------------------
           GMDinstance.init_window_expand_slide('slide',ntrainsampover2,ntrainsampover2);
           
           % make the joint distribution ----------------------------------
           % Gaussian copula, time adaptive
%            JDinstance = JD_GCTAEF(); 
%            JDinstance.input_lambda(0.99);
           % Gaussian copula, exponential covariance
%            JDinstance = JD_GCECMHF(); 
%            JDinstance.input_cov_eps(1e-3); % covariance parameter 
           % Gaussian copula, empricial covariance
%            JDinstance = JD_GCEMP();
            % Gaussian copula independent
            JDinstance = JD_GCIND(); 
           
           MDarray = cell(1,M);
%            % QR:
%            for ii=1:M
%                md = MD_QUANTILE_REGRESSION('c');
%                MDarray{ii} = md;
%            end
%            MD_training = [];
           % KDE:
           SK = KDE_SAVEDKERNEL();
           SK = SK.load_beta();
           for ii=1:M
               assert(~do_error)
               md = MD_KDE();
               k1 = KDEK_BETA(0,1); % response kernel
               if predvars==1
                   k2 = KDEK_BETA(0,1); % predictor kernel
                   md.input_predictor_kernels({k2});
               elseif predvars==2
                   k21 = KDEK_BETA(0,1); 
                   k22 = KDEK_BETA(0,1);
                   md.input_predictor_kernels({k21,k22});
               end
               md.input_response_kernel(k1);
               md.input_SavedKernel(SK);
               % retrain the marginal distribution every so often
               nts_train_every = 24*7; % # time steps
               md.input_train_every(nts_train_every);
               % weight the data points 
               lambda = (0.01)^(1/ntrainsampover2);
               md.input_data_weights_exp(lambda,ntrainsampover2);
               % specify the training objective function
               md.input_training_method('lik'); % either 'cdf' or 'lik'
               MDarray{ii} = md;
           end
           MD_training = {[], Inf}; % (see MD_KDE.get_training for details)
           
           JDinstance.input_MDarray(MDarray);
           JDinstance.input_zero_one_percentiles(zopcell);
           
           %---------------------------------------------------------------
           GS = GEN_SCENARIOS();
           GS.input(JDinstance,GMDinstance);
           
           % NOTE: right now this only works with KDE, not QR
           % input the initial MD training
           
           GS.input_initial_MD_training(MD_training);
           
           GS.input_T(T);
           d1 = datenum(2013,7,31,0,0,0);
           d2 = datenum(2013,7,31,time_horizon,0,0);
           nscen = 500;
           tic
           [respscen,windscen,success] = GS.generate2(d1,d2,nscen);
           % this method starts generating scenarios at M-T time steps
           % earlier. the scenarios at these initial time steps should be
           % exactly equal to the wind realization, so this is a good check
%            GS.plot_scen(respscen,d1-datenum(0,0,0,M-T,0,0),d2,windscen,do_error)
           
           % respscen and windscen have results from time steps that are
           % not between d1, d2. get rid of excess
           r = M-T+1;
           nrows = round(etime(datevec(d2),datevec(d1))/3600)+1;
           windscen = windscen(r:r+nrows-1,:,:);
           respscen = respscen(r:r+nrows-1,:,:);
           % plot this to double check we did things right
           GS.plot_scen(respscen,d1,d2,windscen,do_error)
           toc
           tic
%            results_struct = [];
           % now do error analysis
           if success
               % need to get the wind observation from dates d1 to d2
               obs = DHinstance.get_seq_data_inclusive(d1,d2);
               
               % include a burn-in period - the first couple of scenarios
               % are all conditioned on the same historical previous data
               burn_in = 24; % number of time steps
               results_struct = RUN_GEN_SCENARIOS.analyze_scenarios(...
                   windscen(burn_in+1:end,:,:),obs(burn_in+1:end));
               
           end
           toc
           
           % make a description 
           m = MDarray{1};
           description = struct(...
               'generation_method','beta_analysis',...
               'data_source',data_source,...
               'nts_train_every',nts_train_every,...
               'horizon',horizon,...
               'time_horizon',time_horizon,...
               'ntrainsampover2',ntrainsampover2,...
               'M',M,...
               'T',T,...
               'MD',m.get_description(),...
               'JD',JDinstance.get_description(),...
               'd1',datestr(d1),...
               'd2',datestr(d2),...
               'nscen',nscen);
           for ii=1:size(MPRDarr,1)
               m = MPRDarr{ii,1};
               description.(['MPRDarr_',num2str(ii)]) = m.get_description();
           end
           
           
        end
        
        
       function [GS,respscen,windscen,success] = test2_sides_prev()
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           % make the marginal distributions -----------------------------
           % make 3 marginal distributions and combine up to two MPRDs
           M = 3;
           MPRDarr = cell(2,M);
           do_e = cell(1,M);
           % 1. lead time = 1, 
           do_error = false;
           MPRDarr{1,1} = MPRD_SIDES(1,0,0,do_error);
           MPRDarr{2,1} = MPRD_PREVWIND(1,1,do_error);
           do_e{1} = do_error;
           % also make special basis functions 
           % a linear combination of possibly non-linear basis functions 
           qrb = QRB_LINCOMBO(); 
           qrbf_cell = cell(2);
           qrbf_cell{1} = QRBF_LINEAR(); % linear function of MPRD_SIDES
           sig = QRBF_SIG(); % sigmoidal function of MPRD_PREVWIND
           % sig takes three paramters, 
%            sig.make_constant_b(1);
%            sig.make_constant_b(3);
           qrbf_cell{2} = sig; 
           qrb.input_qrbf_cell(qrbf_cell);
           % 2. lead time = 2
           do_error = true;
           MPRDarr{1,2} = MPRD_SIDES(2,-1,1,do_error);
           do_e{2} = do_error;
           % 3. lead time = 3
           do_error = true;
           MPRDarr{1,3} = MPRD_SIDES(3,-1,1,do_error);
           do_e{3} = do_error;
           % input
           GMDinstance.input_MPRDarray(MPRDarr);
           zopcell = GMDinstance.get_zero_one_percentiles();
           % make the real-world conversions ------------------------------
           CR2Rarr = cell(1,M);
           for ii=1:M
               CR2Rarr{ii} = CR2R_TRAIL(ii,do_e{ii});
           end
           GMDinstance.input_CR2Rarray(CR2Rarr);
           % make the data handler ----------------------------------------
           DHinstance = DATA_HANDLER();
           % input
           GMDinstance.input_DH(DHinstance);
           % initialize the method of getting data ------------------------
           GMDinstance.init_window_expand_slide('expand',24*30,24*30);
           
           %---------------------------------------------------------------
           JDinstance = JD_GCTAEF();
           MDarray = cell(1,M);
           for ii=1:M
               qr = MD_QUANTILE_REGRESSION('c');
               MDarray{ii} = qr;
           end
           JDinstance.input_lambda(0.99);
           JDinstance.input_MDarray(MDarray);
           JDinstance.input_zero_one_percentiles(zopcell);
           
           %---------------------------------------------------------------
           GS = GEN_SCENARIOS();
           GS.input(JDinstance,GMDinstance);
           GS.input_T(1);
           d1 = datenum(2013,6,1,1,0,0);
           d2 = datenum(2013,6,1,6,0,0);
           nscen = 20;
           [respscen,windscen,success] = GS.generate(d1,d2,nscen);
           GS.plot_scen(respscen,d1,d2,windscen)
       end
    end
    
end