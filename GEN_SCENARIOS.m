classdef GEN_SCENARIOS < handle
   % A class used to generate long-term wind realization scenarios
   % conditioned on a set of sequential forecasts.
   properties
       JD % JOINT_DISTRIBUTION instance
       GMD % GET_MARG_DATA instance
       T
       MD_train_init
   end
   methods
       function obj = GEN_SCENARIOS()
       end
       function obj = input(obj,JDinstance,GMDinstance)
           obj.JD  = JDinstance;
           obj.GMD = GMDinstance;
%            obj.GMD.input_DH(DATA_HANDLER());
       end
       function obj = input_T(obj,T)
           % T is the number of wind scenario time steps that should be
           % generated at each iteration
           % e.g. at time step cur_date, wind scenarios should be generated
           % over time steps cur_date, cur_date+1, ..., cur_date+(T-1)
           obj.T = T;
       end
       function obj = input_initial_MD_training(obj,MD_training)
           obj.MD_train_init = MD_training;
       end
       function plot_scen(obj,respscen,start_date,end_date,windscen,do_error)
           if do_error
               do_error = 1;
           else
               do_error = 0;
           end
           % get the wind realization
           obj.GMD.DH.init_constant_window();
%            h = datenum(0,0,0,1,0,0);
           D = obj.GMD.DH.get_data(start_date-datenum(0,0,0,1,0,0),end_date);
           r = D(2:end,1);
           f = D(1:end-1,2);
%            [f,r]'
           figure()
           hold all
           plot(r,'*-r')
           plot(f,'*-k')
           windscen = reshape(windscen,size(windscen,1),size(windscen,3));
           size(windscen)
           size(r)
           plot(windscen(1:size(r,1),:))%,'b')
           plot(r,'*-r','Linewidth',3,'Markersize',5)
           plot(f,'*-k','Linewidth',3,'Markersize',5)
           legend('realization','forecast','scenarios','Location','Best')
           figure()
           hold all
           plot(r,'*-r')
           plot(f,'*-k')
           respscen = reshape(respscen,size(respscen,1),size(respscen,3));
           respscen = respscen(1:size(r,1),:);
           respscen = respscen+do_error*f*ones(1,size(respscen,2));
           plot(respscen)%,'b')
           legend('realization','forecast','scenarios','Location','Best')
           figure()
           hold on
%            windscen = reshape(windscen,size(windscen,1),size(windscen,3));
           windscen = windscen(1:size(r,1),:);
           plot(windscen-respscen)       
       end
       function [respscen,scen,success] = ...
               generateLTG(obj,start_date,end_date,nscen,MDtypelist)
           % generate long-term wind scenarios
           % this function uses the following method:
           %  - At current time curdate create marginal distributions for
           %    wind realizations over time steps cur_date+ii, ii=T-M:T,
           %    where M>T is the number of marginal distributions
           %  - Create joint distribution of the marginals
           %  - Conditioned on the scenarios over time steps curdate+ii,
           %    ii=T-M:0 generate the scenario realization at time steps
           %    curdate+ii, ii=1:T
           %  - Advance T time steps
           % NOTE: marginal distributions 
           success = true;
           % there are M marginal distributions in the joint distribution
           M = obj.GMD.get_num_marg_dist();
           offset = M-obj.T; 
           assert(offset>0)
           nhours = floor(etime(datevec(end_date),datevec(start_date))/3600)+offset+1;
           sd = 1; % number of dimensions in each scenario
           respscen = zeros(nhours,sd,nscen); % response variable scnearios
           scen     = zeros(nhours,sd,nscen); % wind realization scenarios
           % start one time step behind - we want to begin making
           % predictions (scenarios) at start_date
           start_date = start_date - datenum(0,0,0,1,0,0);

           % need to initialize respscen with historical data
           % the first offset time steps of respscen are the response
           % variables from GMD
           [margcellXy,~] = obj.GMD.get_predresp_custom(start_date,start_date);
           for ii=1:offset
               tmp = margcellXy{ii};
               resp = tmp(:,end); % response variables
               for s=1:nscen 
                   respscen(ii,:,s) = resp;
               end
           end
           
           % we can also initialize scen (even though this data is never
           % used). scen is initialized with the wind realizations
           [r,~] = obj.GMD.DH.get_seq_data_inclusive(start_date-datenum(0,0,0,offset-1,0,0),start_date);
           for s=1:nscen
               scen(1:offset,:,s) = r;
           end
           
           cur_MD_train = obj.MD_train_init;
           myiter = 1;

           % get the predictor and response variables 
           [margcellXy, update] = obj.GMD.get_predresp(start_date);
           % add this data
           obj.JD.input_data(margcellXy,update);
           % train the joint and marginal distributions
           obj.JD.train(cur_MD_train,MDtypelist,[]);
           % get the predictor variables given the response variable
           % scenarios that have already been generated
           [margcellX, ~] = ...
               obj.GMD.get_pred_given_resp(start_date,respscen(1:myiter-1,:,:),myiter-1);
           rsprev = respscen(myiter:myiter+offset-1,:,:);
           rsprev = reshape(rsprev,size(rsprev,1),nscen)';
           respscentmp = obj.JD.generate_conditional(margcellX,rsprev);
           respscen(myiter+offset:myiter+offset+obj.T-1,:,:) = ...
                           obj.reshape_2scen(respscentmp);
           % respscentmp only has T columns (time steps), whereas there
           % are M marginal distributions. 
           % GMD.get_scen_given_resp_and_scen assumes respscentmp has M
           % columns. So, need to pad respscentmp - the columns of
           % respscentmp correspond to the last T marginal
           % distributions, so pad respscentmp on the front
           scentmp = obj.GMD.get_scen_given_resp_and_scen(...
               start_date,respscen(1:myiter-1,:,:),myiter-1, [ zeros(nscen,offset), respscentmp ]);
           % conversely, the first offset rows of scentmp now
           % correspond to the marginal distributions that we don't
           % need
           scentmp = scentmp(offset+1:end,:,:);
           scen(myiter+offset:myiter++offset+obj.T-1,:,:) = scentmp;

           respscen = respscen(1:nhours,:,:);
           scen = scen(1:nhours,:,:);
           
       end
       function [respscen,scen,success] = ...
               generateNCM(obj,start_date,end_date,nscen)
           % generate long-term wind scenarios
           % this function uses the following method:
           %  - At current time curdate create marginal distributions for
           %    wind realization over leadtimes 1:T, for T>=1
           %  - Create joint distribution of the marginals
           %  - Generate nscen wind scenarios of duration T and concatenate
           %    these onto the previously generated sequences
           %  - Advance curtime to curtime+(T hours)
           success = true;
           nhours = floor(etime(datevec(end_date),datevec(start_date))/3600)+1;
           % start one time step behind - we want to begin making
           % predictions (scenarios) at start_date
           start_date = start_date - datenum(0,0,0,1,0,0);
           sd = 1; % number of dimensions in each scenario
           respscen = zeros(nhours,sd,nscen); % response variable scnearios
           scen     = zeros(nhours,sd,nscen); % wind realization scenarios
           cur_date = start_date;
           % the marginal distributions are used to create scenarios from 1
           % to M time steps in the future
           % however, the default is to generate scenarios of length
           % 1<=T<=M
           M = obj.GMD.get_num_marg_dist();
           % start the main loop
           scenrow=1;
           hasmadeanyscen=false; % no scenarios have been generated so far
           haspred = true;
           
           last_trained_MD = obj.T;
           is_MD_train_init = true;
           while cur_date < end_date
               % fix Matlab precision issues
               cur_date = obj.round_datenum(cur_date); 
               datestr(cur_date)

               % get the predictor and response variables 
               [margcellXy, update] = obj.GMD.get_predresp(cur_date);
               % add this data
               obj.JD.input_data(margcellXy,update);
               
               % get the previous trainings
               if is_MD_train_init
%                    init_MD_training = obj.MD_train_init;
%                    MDtypelist = 1:length(obj.JD.MDarray);
%                    prev_MD_training_cell = [];
%                    is_MD_train_init = false;
                   
                   init_MD_training = [];
                   MDtypelist = [];
                   prev_MD_training_cell = cell(1,length(obj.JD.MDarray));
                   for ii=1:length(obj.JD.MDarray)
                       prev_MD_training_cell{ii} = obj.MD_train_init;
                   end
                   is_MD_train_init = false;
               else
                   init_MD_training = [];
                   MDtypelist = [];
               end
               
               % determine how many time steps we need to generate
               numdist = obj.T;
               % check if the predictor variables exist
               tmpdate = cur_date+datenum(0,0,0,numdist,0,0);
               [~, haspred] = ...
                   obj.GMD.get_pred_given_resp(tmpdate,respscen(1:scenrow-1,:,:),scenrow-1);
               while ~haspred
                   numdist = numdist+1;
                   if numdist>M
                       error('too many')
                   end
                   tmpdate = obj.round_datenum(tmpdate + datenum(0,0,0,1,0,0));
                   [~, haspred] = ...
                       obj.GMD.get_pred_given_resp(tmpdate,respscen(1:scenrow-1,:,:),scenrow-1);
               end
               disp('-----------')
               numdist
               datestr(tmpdate)
               
               % train the joint and marginal distributions
               obj.JD.train(init_MD_training,MDtypelist,...
                   prev_MD_training_cell,numdist);

               % get the predictor variables given the response variable
               % scenarios that have already been generated
               [margcellX, haspred] = ...
                   obj.GMD.get_pred_given_resp(cur_date,respscen(1:scenrow-1,:,:),scenrow-1);
               % if the predictor variables don't exist then scenarios
               % can't be generated at cur_date
               if haspred
                   % then the predictor variables exist and we're all good
%                    respscentmp =  obj.JD.generate(margcellX);
%                    respscen(scenrow:scenrow+M-1,:,:) = ...
%                                    obj.reshape_2scen(respscentmp);
%                    scentmp = obj.GMD.get_scen_given_resp_and_scen(...
%                        cur_date,respscen(1:scenrow-1,:,:),scenrow-1,respscentmp);
%                    scen(scenrow:scenrow+M-1,:,:) = scentmp;
%                    scenrow = scenrow+obj.T;
%                    cur_date = cur_date + datenum(0,0,0,obj.T,0,0);
                   respscentmp =  obj.JD.generate(margcellX,numdist);
                   respscen(scenrow:scenrow+numdist-1,:,:) = ...
                                   obj.reshape_2scen(respscentmp);
                   scentmp = obj.GMD.get_scen_given_resp_and_scen(...
                       cur_date,respscen(1:scenrow-1,:,:),scenrow-1,respscentmp);
                   scen(scenrow:scenrow+numdist-1,:,:) = scentmp;
                   scenrow = scenrow+numdist;
                   cur_date = cur_date + datenum(0,0,0,numdist,0,0);
%                    hasmadeanyscen = true;
%                    triesremaining = M-obj.T;
%                    last_trained_MD = obj.T;
                   
                   prev_MD_training_cell = cell(1,length(obj.JD.MDarray));
                   for ii=1:length(obj.JD.MDarray)
                       md = obj.JD.MDarray{ii};
                       tr = md.get_training();
                       tr.nts_since_last_trained = ...
                           tr.nts_since_last_trained + obj.T - 1;
                       prev_MD_training_cell{ii} = tr;
                   end
               else
                   error('huh?')
                   % we need to advance one time step and try again
                   scenrow = scenrow+1;
                   cur_date = cur_date + datenum(0,0,0,1,0,0);
                   if hasmadeanyscen
                       % we need to be careful since the last scenarios
                       % that were generated were of length M
                       % this gives us a total of M-T attempts to advance
                       % to the next time step and hope that predictor
                       % variables can be created
                       % otherwise there will be a gap in scenario
                       % generation
                       if triesremaining<=0
                           error('do not have predictor variables')
                       else
                           triesremaining = triesremaining-1;
                           last_trained_MD = last_trained_MD+1;
                       end
                   end
                   for ii=1:length(obj.JD.MDarray)
                       tr = prev_MD_training_cell{ii};
                       tr.nts_since_last_trained = ...
                           tr.nts_since_last_trained + 1;
                       prev_MD_training_cell{ii} = tr;
                   end
               end
           end
       end
       function scen = generateBPA(obj,start_date,niters,nhours,nscen)
           % generate scenarios used for BPA control problem
           % inputs:
           %  - start_date: datenum, starting hour
           %  - niters: number of iterations 
           %  - nhours: number of hours in each stage
           %  - nscen: positive integer, number of scenarios to generate
           % returns:
           %  - scen: nscen X 72 X niters matrix
           % algorithm:
           %  At iteration kk=1:niters
           %    select wind forecast f_kk generated at time
           %      t0_kk start_date+hours(nhours*(kk-1))
           %    use this forecast to generate nscen wind scenarios over 
           %      hours t0_kk+1 to t0_kk+72 conditioned on forecast f_kk
           %    set these scenarios, contained in a nscenX72 matrix, into
           %      scen(:,:,kk)
           %  NOTE: the predictor variables must exist, i.e. fill in
           %        missing historical forecasts with persistence

           % start one time step behind - we want to begin making
           % predictions (scenarios) at start_date
           start_date = start_date - datenum(0,0,0,1,0,0);
           % length (in hours) of each scenario
           M = 72;
           scen = zeros(nscen,M,niters); % wind realization scenarios
           
           is_MD_train_init = true;
           for kk = 1:niters
               cur_date = obj.round_datenum(start_date+datenum(0,0,0,nhours*(kk-1),0,0));
               datestr(cur_date)

               % get the predictor and response variables 
               [margcellXy, update] = obj.GMD.get_predresp(cur_date);
               % add this data
               obj.JD.input_data(margcellXy,update);
               
               % get the previous trainings
               if is_MD_train_init
                   init_MD_training = [];
                   MDtypelist = [];
                   prev_MD_training_cell = cell(1,length(obj.JD.MDarray));
                   for ii=1:length(obj.JD.MDarray)
                       prev_MD_training_cell{ii} = obj.MD_train_init;
                   end
                   is_MD_train_init = false;
               else
                   init_MD_training = [];
                   MDtypelist = [];
               end

               % check that the forecast exists
               [~, haspred] = ...
                   obj.GMD.get_pred_given_resp(cur_date,zeros(0,1,nscen),0);
               if ~haspred
                   error(['forecast does not exist: ',datestr(cur_date)])
               end
               
               % train the joint and marginal distributions
               obj.JD.train(init_MD_training,MDtypelist,...
                   prev_MD_training_cell,M);

               % get the predictor variables given the response variable
               % scenarios that have already been generated
               [margcellX, ~] = ...
                   obj.GMD.get_pred_given_resp(cur_date,zeros(0,1,nscen),0);
               respscentmp =  obj.JD.generate(margcellX,M);
               scentmp = obj.GMD.get_scen_given_resp_and_scen(...
                   cur_date,zeros(0,1,nscen),0,respscentmp);
               size(scentmp)
               scen(:,:,kk) = reshape(scentmp,72,nscen)';

               prev_MD_training_cell = cell(1,length(obj.JD.MDarray));
               for ii=1:length(obj.JD.MDarray)
                   md = obj.JD.MDarray{ii};
                   tr = md.get_training();
                   tr.nts_since_last_trained = ...
                       tr.nts_since_last_trained + M - 1;
                   prev_MD_training_cell{ii} = tr;
               end
               
           end
       end
   end
   methods (Static, Access=protected)
       function scen = reshape_2scen(R)
           % R is response variables of size SXM
           % scen is of size MX1XS
           [S,M] = size(R);
           scen = reshape(R',M,1,S);
       end
       
   end
   methods (Static)
       function date = round_datenum(date)
           % Matlab dates are buggy (at least in 2012)
           % there are some precision issues:
           % >> a=datenum(2013,6,1,0,0,0);
           % >> for ii=1:24
           %        a=a+datenum(0,0,0,1,0,0);
           %    end
           % >> a==datenum(2013,6,2,0,0)
           %    ans = 0
           y = year(date);
           m = month(date);
           d = day(date);
           h = hour(date);
           diff = zeros(1,3);
           for ii=-1:1
               diff(ii+2) = abs(datenum(y,m,d,h+ii,0,0)-date);
           end
           ii = find(diff==min(diff));
           date = datenum(y,m,d,h+ii-2,0,0);
       end
       function [respscen,windscen,success] = test()
           %---------------------------------------------------------------
           GMDinstance = GET_MARG_DATA();
           % make the marginal distributions -----------------------------
           % make 3 marginal distributions and combine up to two MPRDs
           M = 3;
           MPRDarr = cell(2,M);
           % 1. lead time = 1, 
           MPRDarr{1,1} = MPRD_SIDES(1,0,0);
           % 2. lead time = 2
           MPRDarr{1,2} = MPRD_SIDES(2,-1,-1);
           MPRDarr{2,2} = MPRD_SIDES(2,0,1);
           % 3. lead time = 3
           MPRDarr{1,3} = MPRD_SIDES(3,1,1);
           % input
           GMDinstance.input_MPRDarray(MPRDarr);
           zopcell = GMDinstance.get_zero_one_percentiles();
           % make the real-world conversions ------------------------------
           CR2Rarr = cell(1,M);
           for ii=1:M
               CR2Rarr{ii} = CR2R_TRAIL(1);
           end
           GMDinstance.input_CR2Rarray(CR2Rarr);
           % make the data handler ----------------------------------------
           DHinstance = DATA_HANDLER();
           % input
           GMDinstance.input_DH(DHinstance);
           % initialize the method of getting data ------------------------
           GMDinstance.init_window_expand_slide('expand',24*14,0);
           
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
           d1 = datenum(2013,6,1,0,0,0);
           d2 = datenum(2013,6,1,18,0,0);
           nscen = 10;
           [respscen,windscen,success] = GS.generate(d1,d2,nscen);
           GS.plot_scen(respscen,d1,d2,windscen)
       end
   end
end