#!/usr/bin/env python
from energy_force import *

def model(s):
	A_eps 	= s["Parameters"][0]
	B		= s["Parameters"][1]
	sigma	= s["Parameters"][2]
	rc		= s["Parameters"][3]

	sig = s["Parameters"][4]
	
	U_e, F = energy_force(A_eps * 6.947695457055374e-21, B, sigma, rc)


	result = [U_e, 1e9*F]
	sdev   = [sig*abs(U_e), sig*abs(1e9*F)]
	
	s["Reference Evaluations"] = result
	s["Standard Deviation"]    = sdev


def model2(s):
	C 	  = s["Parameters"][0]
	D	  = s["Parameters"][1]
	sigma = s["Parameters"][2]
	rc    = s["Parameters"][3]

	sig = s["Parameters"][4]
	
	U_e, F = energy_force2(C * 6.947695457055374e-21, D * 6.947695457055374e-21, sigma, rc)


	result = [U_e, 1e8*F]
	sdev   = [sig*abs(U_e), sig*abs(1e8*F)]
	
	s["Reference Evaluations"] = result
	s["Standard Deviation"]    = sdev


def model_lj(s):
	eps   = s["Parameters"][0]
	sigma = s["Parameters"][1]
	rc    = s["Parameters"][2]

	sig = s["Parameters"][3]
	
	U_e, F = energy_force_lj(eps * 6.947695457055374e-21, sigma, rc)

	F *= 1e9	#1e9	amplify magnitude of F
	
	result = [U_e, F]
	sdev   = [sig*abs(U_e), sig*abs(F)]
	
	s["Reference Evaluations"] = result
	s["Standard Deviation"]    = sdev
  


def getReferenceData():
  y=[]
  y.append(260.0)	#Adhesion Energy 			[mJ/m^2]
  y.append(0.0)		#Total force in z direction	[N]
  return y
