from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import pandas as pd
from sumolib import checkBinary  # noqa
import traci  # noqa


def generate_routefile(pWE=0.05,pNS=0.15):
    random.seed(42)  # make tests reproducible可重复
    N = 6000
    with open("routfile/time_vary_test.rou.xml", "w") as routes:
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
            if 0 <= i < 1000:
                pWE = 0.05
                pNS = 0.3
            if 1000 <= i < 2000:
                pWE = 0
                pNS = 0 
            if 2000 <= i < 3000:
                pWE = 0.05
                pNS = 0.05
            if 3000 <= i < 4000:
                pWE = 0
                pNS = 0 
            if 4000 <= i < 5000:
                pWE = 0.3
                pNS = 0.05
            if 5000 <= i < 6000:
                pWE = 0
                pNS = 0
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="NS_%i" type="typeStraight" route="NS" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="NE_%i" type="typeTurn" route="NE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="SN_%i" type="typeStraight" route="SN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="SW_%i" type="typeTurn" route="SW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="WE_%i" type="typeStraight" route="WE" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="WN_%i" type="typeTurn" route="WN" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="EW_%i" type="typeStraight" route="EW" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="ES_%i" type="typeTurn" route="ES" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

if __name__ == "__main__":
    generate_routefile()
    print("Completed!")