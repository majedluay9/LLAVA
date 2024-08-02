import sklearn
import time

import utils
from csv_processor import *

from os import listdir
from os.path import isfile, join

###################################### Demo embedding on a static dataset

DIR = '/afm01/Q1/Q1002/comscentre/CSV_files/split/pecv/'
RES_SMALL = 60 # Temporal resolution of averaging (in seconds)
RES_BIG = 60 # Number of temporal resolution units per timestep
FNAME = DIR + 'split_000000'
KERNEL_LIMIT = 1000#np.inf

###################### Instantiate csv processors for IPs and Ports

csv_processor_ip = CsvProcessor(FNAME, idx=[H_IP_SRC, H_IP_DST])
#csv_processor_ip.plot_by_resolution(RES_SMALL*RES_BIG, name='ip')
data_ip = csv_processor_ip.get_sparse_array(RES_SMALL, RES_BIG)

csv_processor_port = CsvProcessor(FNAME, idx=[H_PORT_SRC, H_PORT_DST])
#csv_processor_port.plot_by_resolution(RES_SMALL*RES_BIG, name='port')
data_port = csv_processor_port.get_sparse_array(RES_SMALL, RES_BIG)


data_ip_av = sp.csr_matrix((0,0))
data_port_av = sp.csr_matrix((0,0))
#for f in files[1:]:
for i in range (0, 41064):
    f = 'split_' + str(i).zfill(6)
    t0 = time.time()
    print(f)
    f = DIR + f

    data_ip = csv_processor_ip.add_new_data(f, data_ip, RES_SMALL, RES_BIG)
    data_port = csv_processor_port.add_new_data(f, data_port, RES_SMALL, RES_BIG)
    
    n_features_ip   = csv_processor_ip.num_nodes**2*len(csv_processor_ip.attributes)
    n_features_port = csv_processor_port.num_nodes**2*len(csv_processor_port.attributes)

    ## Embed into low dimensional space using kPCA
    data_ip_dropped = utils.drop_sparse_cols(data_ip)
    data_port_dropped = utils.drop_sparse_cols(data_port)

    data_ip_av = utils.update_moving_average(data_ip_dropped, data_ip_av, RES_BIG)
    data_port_av = utils.update_moving_average(data_port_dropped, data_port_av, RES_BIG)
   
    #data_ip_av = utils.moving_average(data_ip_dropped, RES_BIG)
    #data_port_av = utils.moving_average(data_port_dropped, RES_BIG)
    t1 = time.time()
    print("Took " + str(t1-t0) + " seconds for reading data and averaging")
    if data_ip_av.shape[0] > 1:
        print(data_ip_av.shape[0])
        t0 = time.time()
        input_data_ip = sklearn.preprocessing.scale(data_ip_av, with_mean=False)
        input_data_port = sklearn.preprocessing.scale(data_port_av, with_mean=False)

        data_all = sp.hstack((input_data_ip, input_data_port))

        kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
                gamma=1/(n_features_ip))
        kpca.fit(utils.remove_old_data(input_data_ip, RES_BIG, KERNEL_LIMIT))
        transformed = kpca.transform(input_data_ip)
        t1 = time.time()
        print("Took " + str(t1-t0) + " seconds for kPCA")

        t0 = time.time()
        utils.plot_transformed(transformed, csv_processor_ip.start_time, 
                RES_SMALL, f+'embedding_ip.png')
        t1 = time.time()
        print("Took " + str(t1-t0) + " seconds for plotting")
        
        kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
                gamma=1/(n_features_port))
        kpca.fit(utils.remove_old_data(input_data_port, RES_BIG, KERNEL_LIMIT))
        transformed = kpca.transform(input_data_port)
        utils.plot_transformed(transformed, csv_processor_port.start_time, 
                RES_SMALL, f+'embedding_port.png')
        print("done embedding")

        kpca = sklearn.decomposition.KernelPCA(n_components=2, kernel='rbf', 
                gamma=1/((n_features_ip+n_features_port)))
        kpca.fit(utils.remove_old_data(data_all, RES_BIG, KERNEL_LIMIT))
        transformed = kpca.transform(data_all)
        utils.plot_transformed(transformed, csv_processor_ip.start_time, 
                RES_SMALL, f+'embedding_all.png')
        print("done embedding")

