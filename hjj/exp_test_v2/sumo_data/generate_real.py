from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import pandas as pd
from sumolib import checkBinary  # noqa
import traci  # noqa

def setEWNS(pE, pW, pN, pS):
    pE = pE/3600 + 0.1
    pW = pW/3600 + 0.1
    pN = pN/3600  
    pS = pS/3600 
    return pE, pW, pN, pS

def generate_routefile(pE=0,pW=0,pN=0,pS=0):
    random.seed(42)  # make tests reproducible可重复
    N = 86400
    with open("/home/hjj/exp_test_v2/sumo_data/routfile/real.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="typeStraight" accel="1" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.9" guiShape="passenger"/>
        <vType id="typeTurn" accel="1" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.9" guiShape="passenger"/>

        <route id="WE" edges="1i 2o" />
        <route id="EW" edges="2i 1o " />
        <route id="NS" edges="4i 3o " />
        <route id="SN" edges="3i 4o " />
        <route id="NE" edges="4i 2o" />
        <route id="ES" edges="2i 3o " />
        <route id="SW" edges="3i 1o " />
        <route id="WN" edges="1i 4o " />""", file=routes)
        vehNr = 0
        

        for i in range(N):
            if 0 <= i < 3600*1:
                pE, pW, pN, pS=setEWNS(150, 200, 50, 280)
            if 3600 <= i < 3600*2:
                pE, pW, pN, pS=setEWNS(60, 60, 30, 80)
            if 3600*2 <= i < 3600*3:
                pE, pW, pN, pS=setEWNS(20, 30, 25, 30)
            if 3600*3 <= i < 3600*4:
                pE, pW, pN, pS=setEWNS(25, 35, 30, 35)  
            if 3600*4 <= i < 3600*5:
                pE, pW, pN, pS=setEWNS(35, 35, 35, 40)
            if 3600*5 <= i < 3600*6:
                pE, pW, pN, pS=setEWNS(80, 70, 70, 100)
            if 3600*6 <= i < 3600*7:
                pE, pW, pN, pS=setEWNS(250, 200, 150, 250)
            if 3600*7 <= i < 3600*8:
                pE, pW, pN, pS=setEWNS(500, 220, 180, 200)  
            if 3600*8 <= i < 3600*9:
                pE, pW, pN, pS=setEWNS(750, 50, 200, 50)  
            if 3600*9 <= i < 3600*10:
                pE, pW, pN, pS=setEWNS(900, 100, 250, 100)
            if 3600*10 <= i < 3600*11:
                pE, pW, pN, pS=setEWNS(700, 200, 220, 500)
            if 3600*11 <= i < 3600*12:
                pE, pW, pN, pS=setEWNS(750, 450, 210, 550)  
            if 3600*12 <= i < 3600*13:
                pE, pW, pN, pS=setEWNS(600, 350, 200, 400)
            if 3600*13 <= i < 3600*14:
                pE, pW, pN, pS=setEWNS(1000, 500, 300, 700)
            if 3600*14 <= i < 3600*15:
                pE, pW, pN, pS=setEWNS(700, 450, 300, 650)
            if 3600*15 <= i < 3600*16:
                pE, pW, pN, pS=setEWNS(650, 420, 280, 600)  
            if 3600*16 <= i < 3600*17:
                pE, pW, pN, pS=setEWNS(630, 460, 260, 580) 
            if 3600*17 <= i < 3600*18:
                pE, pW, pN, pS=setEWNS(950, 550, 270, 750)
            if 3600*18 <= i < 3600*19:
                pE, pW, pN, pS=setEWNS(500, 500, 250, 650)
            if 3600*19 <= i < 3600*20:
                pE, pW, pN, pS=setEWNS(750, 450, 300, 550)  
            if 3600*20 <= i < 3600*21:
                pE, pW, pN, pS=setEWNS(500, 500, 400, 540)
            if 3600*21 <= i < 3600*22:
                pE, pW, pN, pS=setEWNS(300, 470, 200, 560)
            if 3600*22 <= i < 3600*23:
                pE, pW, pN, pS=setEWNS(250, 420, 200, 450)
            if 3600*23 <= i < 3600*24:
                pE, pW, pN, pS=setEWNS(240, 320, 180, 350)   

            if random.uniform(0, 1) < pN:
                print('    <vehicle id="NS_%i" type="typeStraight" route="NS" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pN:
                print('    <vehicle id="NE_%i" type="typeTurn" route="NE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pS:
                print('    <vehicle id="SN_%i" type="typeStraight" route="SN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pS:
                print('    <vehicle id="SW_%i" type="typeTurn" route="SW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pW:
                print('    <vehicle id="WE_%i" type="typeStraight" route="WE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pW:
                print('    <vehicle id="WN_%i" type="typeTurn" route="WN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pE:
                print('    <vehicle id="EW_%i" type="typeStraight" route="EW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pE:
                print('    <vehicle id="ES_%i" type="typeTurn" route="ES" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

if __name__ == "__main__":
    generate_routefile()
    print("Completed!")