
import os
import pandas as pd
import matplotlib
import numpy as np

import PARAMETERS
from GLOBAL_Parameters import *


class DataScaling(object):
    def __init__(self, fname,
                 attributes=[H_IN_PKTS, H_OUT_PKTS, H_IN_BYTES, H_OUT_BYTES]):
        self.fname = fname
        # READ THE FILE INTO A PANDAS DATAFRAME
        self.df = pd.read_csv(fname)
        self.attributes = attributes
        self.start_time = self.df[H_LAST_SWITCHED].min()
        self.end_time = self.df[H_LAST_SWITCHED].max()
        # self.log_transform_attributes()                                                                                   # I need to put numerical features into log + 1 scale

    def log_transform_attributes(self):
        for attr in self.attributes:
            if attr in [H_IN_PKTS, H_OUT_PKTS, H_IN_BYTES, H_OUT_BYTES]:  # Ensure the column is numeric
                self.df[attr] = self.log_transform_feature(self.df[attr])

    def log_transform_feature(self, data, small_constant=1):
        data = np.array(data)
        transformed_data = np.log(data + small_constant)
        return np.int64(transformed_data)


if __name__ == '__main__':

    FNAME = PARAMETERS.ToN_26Apr
    Heading = 'ToN_26Apr'

    csv = DataScaling(FNAME)
    print(f'Before scalling:\n{csv.df[csv.attributes].mean()}')
    csv.log_transform_attributes()
    print(f'After scalling:\n{csv.df[csv.attributes].mean()}')

    Output_NAME = FNAME[:-4] + '_LogScalled' + FNAME[-4:]

    csv.df.to_csv(Output_NAME)
    print('Done')