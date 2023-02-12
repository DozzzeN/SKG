#
# Copyright (C) 2019 Luca Pasqualini
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
# Inspired by the work of David Johnston (C) 2017: https://github.com/dj-on-github/sp800_22_tests
#
# NistRng is licensed under a BSD 3-Clause.
#
# You should have received a copy of the license along with this
# work. If not, see <https://opensource.org/licenses/BSD-3-Clause>.

# Import packages
import csv

import numpy

# Import src
import numpy as np

from nistrng import *

import fileinput

if __name__ == "__main__":
    read_data = []
    for line in fileinput.input("dct.txt"):
        for i in range(0, len(line), 20):
            if line[i] == '\n':
                continue
            read_data.append(int(line[i:i + 20]))

    read_data = np.array(read_data)
    # Check the eligibility of the test and generate an eligible battery from the default NIST-sp800-22r1a battery
    eligible_battery: dict = check_eligibility_all_battery(read_data, SP800_22R1A_BATTERY)
    # Print the eligible tests
    print("Eligible test from NIST-SP800-22r1a:")
    for name in eligible_battery.keys():
        print("-" + name)
    # Test the sequence on the eligible tests
    results = run_all_battery(read_data, eligible_battery, False)
    # Print results one by one
    print("Test results:")
    for result, elapsed_time in results:
        if result.passed:
            print("- PASSED - score: " + str(
                numpy.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
        else:
            print("- FAILED - score: " + str(
                numpy.round(result.score, 3)) + " - " + result.name + " - elapsed time: " + str(elapsed_time) + " ms")
