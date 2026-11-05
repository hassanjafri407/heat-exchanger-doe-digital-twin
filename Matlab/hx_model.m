function [Q, Th_out, UA] = hx_model(Th, Tc, Vh, Vc)
A_tube = 9.8169e-5;  A_ann = 8.7666e-5;          
rh = 988; cph = 4181; rc = 999; cpc = 4186;
vh = Vh/60000/A_tube;  vc = Vc/60000/A_ann;      
UA = 1/(0.0003885 + 0.0022158*vh^-0.8 + 0.0022736*vc^-0.8); 
Ch = Vh/60000*rh*cph;  Cc = Vc/60000*rc*cpc;    
Cmin = min(Ch,Cc);  Cmax = max(Ch,Cc);  Cr = Cmin/Cmax;  NTU = UA/Cmin;
if abs(Cr-1) < 1e-9
    eps = NTU/(1+NTU);
else
    eps = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr))); 
end
Q = eps*Cmin*(Th - Tc);
Th_out = Th - Q/Ch;
end




