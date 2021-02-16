# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

"""
Version 2.0
"""

import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.core.numeric import zeros_like

class CClassifier(object):
    def forward(self, z):
        pass

# equal to sigmoid but it is used as classification function
class Logistic(CClassifier):
    def forward(self, z):
        a = 1.0 / (1.0 + np.exp(-z))
        return a

class Softmax(CClassifier):
    def forward(self, z):
        a = zeros_like(z)
        for i in range(z.shape[0]):
            shift_z = z[i,:] - np.max(z[i,:])
            exp_z = np.exp(shift_z)
            a[i,:] = exp_z / np.sum(exp_z)
        #end for
        return a

