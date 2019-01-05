#Top of program#
import math as m
import csv

##########################
class Coordinate:
    def __init__(self, rho, theta, phi, xOff = 0, yOff = 0, zOff = 0, thetaOff = 0, phiOff = 0):
        self.rho = rho
        self.theta = theta - thetaOff
        self.phi = phi - phiOff
        self.xOff = xOff
        self.yOff = yOff
        self.zOff = zOff
    def parameterize(self):
        self.xVal = self.rho*m.cos(self.phi)*m.cos(self.theta) - self.xOff
        self.yVal = self.rho*m.cos(self.phi)*m.sin(self.theta) - self.yOff
        self.zVal = self.rho*m.sin(self.phi) - self.zOff
    def get_xyz(self):
        return([self.xVal,self.yVal,self.zVal])

