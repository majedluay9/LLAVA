import sklearn

import utils
from csv_processor import *

###################################### Demo embedding on a static dataset

FNAME = PARAMETERS.REAL3
RES_SMALL = 60*60 # Temporal resolution of averaging (in seconds)
RES_BIG = 1 # Number of temporal resolution units per timestep

# Instantiate csv processors for IPs and Ports
csv_processor_ip = CsvProcessor(FNAME, idx=[H_IP_SRC, H_IP_DST])
#csv_processor_ip.plot_by_resolution(RES_SMALL*RES_BIG, name='ip')
data_ip = csv_processor_ip.get_sparse_array(RES_SMALL, RES_BIG)

csv_processor_port = CsvProcessor(FNAME, idx=[H_PORT_SRC, H_PORT_DST])
#csv_processor_port.plot_by_resolution(RES_SMALL*RES_BIG, name='port')
data_port = csv_processor_port.get_sparse_array(RES_SMALL, RES_BIG)

n_features_ip   = csv_processor_ip.num_nodes**2*len(csv_processor_ip.attributes)
n_features_port = csv_processor_port.num_nodes**2*len(csv_processor_port.attributes)

## Embed into low dimensional space using kPCA
data_ip = utils.drop_sparse_cols(data_ip)
data_ip = utils.moving_average(data_ip, RES_BIG)
data_port = utils.drop_sparse_cols(data_port)
data_port = utils.moving_average(data_port, RES_BIG)

data_ip = sklearn.preprocessing.scale(data_ip, with_mean=False)
data_port = sklearn.preprocessing.scale(data_port, with_mean=False)
data_all = sp.hstack((data_ip, data_port))

kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
        gamma=1/(n_features_ip))
transformed = kpca.fit_transform(data_ip)
utils.plot_transformed(transformed, csv_processor_ip.start_time, 
        RES_SMALL, 'embedding_ip.png')

kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
        gamma=1/(n_features_port))
transformed = kpca.fit_transform(data_port)
utils.plot_transformed(transformed, csv_processor_port.start_time, 
        RES_SMALL, 'embedding_port.png')

kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
        gamma=1/((n_features_ip+n_features_port)))
transformed = kpca.fit_transform(data_all)
utils.plot_transformed(transformed, csv_processor_ip.start_time, 
        RES_SMALL, 'embedding_all.png')

