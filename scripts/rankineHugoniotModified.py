from numpy import *

Ma = 2.37;
#Ma = 1.25;
#Ma = 1.782;
gamma = 1.4;
Pi = 0

p_preS = 0.85714;
rho_preS = 1.20;
sos_preS = sqrt(gamma * (p_preS+Pi) / rho_preS);
print("sos: ", sos_preS)


p = (1 + 2*gamma/(gamma+1)*(Ma * Ma-1.0)) * p_preS

rho = Ma * Ma/(1 + (gamma-1)/(gamma+1) * (Ma*Ma - 1)) * rho_preS

Vs = Ma * sos_preS

u = Vs * (1 - (1+(gamma-1)/(gamma+1) * (Ma * Ma - 1))/(Ma*Ma))

sos_posS = sqrt(gamma * p / rho);
print(u / sos_posS)
print(u / sos_preS)
print(Vs / sos_posS)
print(Vs / sos_preS)

print("\n\nPre-shock conditions:\n")
print("Ma_s         " + str(Ma) + "\n")
print("rho_preS     " + str(rho_preS))
print("p_preS       " + str(p_preS))
print("u_preS       0")


print("\n\nPost-shock conditions:\n")
print("rho          " + str(rho))
print("p            " + str(p))
print("u            " + str(u))

print("\n\nRatio")
print("rho / rho_preS    " + str(rho / rho_preS))
print("p   / p_preS      " + str(p / p_preS))

print("\n\nShock speed conditions:\n")
print("V_s               " + str(Vs))



