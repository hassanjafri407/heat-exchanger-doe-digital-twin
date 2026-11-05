T = readtable('heat_exchanger_raw_measurements.csv');
T.t_min = minutes(timeofday(datetime(T.time,'InputFormat','HH:mm'))) ...
    - minutes(timeofday(datetime(T.time(1),'InputFormat','HH:mm')));
t   = T.t_min;          Th_meas = T.hot_in_C;
Vh  = T.hot_flow_Lmin;  Vc = T.cold_flow_Lmin;  Tc = T.cold_in_C;
Tset = 55.9;  cp = 4181;
use = 1:15;             

p0 = [30, 5000];       
p  = abs(fminsearch(@(p) sse(p,t,Th_meas,Vh,Vc,Tc,Tset,cp,use), p0));
fprintf('Fitted: M = %.1f kg,  P = %.0f W\n', p(1), p(2));

[tm, Tm] = ode45(@(tq,y) rhs(tq,y,p,t,Vh,Vc,Tc,Tset,cp), [0 t(end)], Th_meas(1));

figure; plot(tm,Tm,'b-',t,Th_meas,'ro','LineWidth',1.5); grid on
xlabel('Time (min)'); ylabel('Hot inlet temperature (°C)');
legend('Model','Measured'); title('Hot reservoir transient');

res = interp1(tm,Tm,t) - Th_meas;
fprintf('RMSE all 16 = %.2f C,  trials 1-15 = %.2f C\n', ...
    sqrt(mean(res.^2)), sqrt(mean(res(use).^2)));

function dT = rhs(tq,T,p,t,Vh,Vc,Tc,Tset,cp)
k = find(t<=tq,1,'last');                    
Q = hx_model(T,Tc(k),Vh(k),Vc(k));           
heat = abs(p(2))*0.5*(1-tanh((T-Tset)/0.6)); 
dT = 60*(heat - Q)/(abs(p(1))*cp);      
end

function e = sse(p,t,Tm,Vh,Vc,Tc,Tset,cp,use)
[ts,Ts] = ode45(@(tq,y) rhs(tq,y,p,t,Vh,Vc,Tc,Tset,cp),[0 t(end)],Tm(1));
e = sum((interp1(ts,Ts,t(use))-Tm(use)).^2);
end