import numpy as np
import networkx as nx
import community
import scanorama
from sklearn.manifold import TSNE
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.cluster import adjusted_rand_score
import os
from scipy import stats

sample_name = sys.argv[1]

def knn(matrix, n_neighbors=20):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(matrix)
    graph = nbrs.kneighbors_graph(matrix).toarray() 
    return graph


def split_embedding():
    Phenotypes = {"Br1416":"Schizo","Br1446":"Control","Br1682":"Schizo","Br1092":"Control","Br2288":"Control",
            "Br1425":"Schizo","Br1245":"Schizo","Br1378":"Control"}
    sample_idx = {"Br1416":0,"Br1446":1,"Br1682":2,"Br1092":3,"Br2288":4,"Br1425":5,"Br1245":6,"Br1378":7}
    sample_mask, bin_all = np.zeros((8, 2534)), []
    infile = open("./datasets/matrix/Schizo_Control/prediction/single_eval/cell_names.txt", "r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        cell_name, sample_name, type_name = line[0], line[1], line[2]
        sample_mask[sample_idx[sample_name],idx] = 1

    sample_rates = []
    for sample_name in ["Br1416","Br1446","Br1682","Br1092","Br2288","Br1425","Br1245","Br1378"]:
        rate_tmp = np.load('./datasets/matrix/Schizo_Control/prediction/100kb_bin/'+sample_name+'/cell_rate.100kbin.mCG.npy')
        sample_rates.append(np.std(rate_tmp[:,:], axis=0))
    sample_rates = np.array(sample_rates)
    bin_variances = np.std(sample_rates, axis=0)
    infile = open("./datasets/matrix/bin_all.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        bin_all.append(line)
    bin_all = np.array(bin_all)
    binlist, data = [], []
    for sample_name in ["Br1416","Br1446","Br1682","Br1092","Br2288","Br1425","Br1245","Br1378"]:
        rate_bin = np.load('./datasets/matrix/Schizo_Control/prediction/100kb_bin/'+sample_name+'/cell_rate.100kbin.mCG.npy')
        cell_meta = np.load('./datasets/matrix/Schizo_Control/prediction/100kb_bin/'+sample_name+'/cell_meta.genome.mCG.npy')
        binfilter = np.logical_and((np.std(rate_bin, axis=0)>0.01), bin_variances<1)#0.05
        print(sample_name, len(rate_bin), sum(binfilter))
        rate_tmp = rate_bin
        rate_tmp = np.divide(rate_tmp[:,binfilter].T, cell_meta.astype(float)).T
        rateb = np.divide((rate_tmp[:,:]-np.mean(rate_tmp[:,:],axis=0)), np.std(rate_tmp[:,:],axis=0).astype(float))
        data.append(rateb)
        binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

    integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
    rateb_reduce = np.concatenate(integrated)
    n_components, ndim = 2, 50
    tsne = TSNE(n_components)
    y = tsne.fit_transform(rateb_reduce[:, :ndim])

    np.save('./datasets/matrix/Schizo_Control/prediction/single_eval/cell_mCG_all_integrated_svd100.npy', rateb_reduce)
    np.savetxt('./datasets/matrix/Schizo_Control/prediction/single_eval/cell_mCG_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')


def split_clustering():
    ndim, cell_names = 100, []
    Phenotypes = {"Br1416":"Schizo","Br1446":"Control","Br1682":"Schizo","Br1092":"Control",
            "Br2288":"Control","Br1425":"Schizo","Br1245":"Schizo","Br1378":"Control"}
    for sample_name in ["Br1416","Br1446","Br1682","Br1092","Br2288","Br1425","Br1245","Br1378"]:
        infile = open("./datasets/matrix/Schizo_Control/prediction/single_eval/"+sample_name+"/cell_names.txt", "r")
        for idx,line in enumerate(infile):
            line = line.strip("\n").split()
            cell_name, sample_name, type_name = line[0], line[1], line[2]
            cell_names.append([cell_name, sample_name, Phenotypes[sample_name], type_name])
    g = knn(np.load('./datasets/matrix/Schizo_Control/prediction/single_eval/cell_mCG_all_integrated_svd100.npy')[:, :ndim], n_neighbors=20)
    inter = g.dot(g.T)
    diag = inter.diagonal()
    jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
    adj = nx.from_numpy_matrix(np.multiply(g, jac)) #.toarray())
    knnjaccluster = {}
    for res in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10,15,20,25,50,100,200,500]:
        partition = community.best_partition(adj,resolution=res)
        label = np.array([k for k in partition.values()])
        knnjaccluster[res] = label
        nc = len(set(label))
        count = np.array([sum(label==i) for i in range(nc)])
        print(res, nc, count)

        infile = open("./datasets/matrix/Schizo_Control/prediction/single_eval/cell_mCG_all_integrated_svd50_p50_rs0.txt", "r")
        output = open("./plot/Schizo_Control/impute_cluster_"+str(res)+"_"+str(nc)+".txt", "w")
        for idx,line in enumerate(infile):
            line = line.strip("\n")
            cell_name,sample_name,phenotype,type_name = cell_names[idx][0],cell_names[idx][1],cell_names[idx][2],cell_names[idx][3]
            output.write(cell_name+"\t"+line+"\t"+"cell"+str(knnjaccluster[res][idx])+"\t"+phenotype+"\t"+type_name+"\t"+sample_name+"\n")
        output.close()
    np.save('./datasets/matrix/Schizo_Control/prediction/single_eval/cell_mCG_all_integrated_svd50_knn20_louvain.npy', knnjaccluster)


def prediction_embedding():
    rate_bin = np.load('./plot/Schizo_Control/prediction/cell_rate.100kbin.mCG.npy')
    read_bin = np.load('./plot/Schizo_Control/prediction/cell_read.100kbin.mCG.npy')
    cell_meta = np.load('./plot/Schizo_Control/prediction/cell_meta.genome.mCG.npy')

    bin_all = []
    infile = open("./datasets/matrix/bin_all.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        bin_all.append(line)
    bin_all = np.array(bin_all)

    binlist, data, read_threshold = [], [], 0
    rate_tmp, read_tmp = rate_bin, read_bin
    s = np.sum(read_tmp>read_threshold, axis=0)
    binfilter = np.logical_and((s>=0.9*len(read_tmp)), bin_all[:,0]!='chrX') #0.9
    print(len(rate_tmp), sum(binfilter))
    rateb = np.divide(rate_tmp[:, binfilter].T, np.max(cell_meta).astype(float)).T
    data.append(rateb)
    binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

    integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
    rateb_reduce = np.concatenate(integrated)
    n_components, ndim = 2, 50
    tsne = TSNE(n_components)
    y = tsne.fit_transform(rateb_reduce[:, :ndim])

    np.save('./plot/Schizo_Control/prediction/cell_mCG_all_integrated_svd100.npy', rateb_reduce)
    np.savetxt('./plot/Schizo_Control/prediction/cell_mCG_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')


def prediction_clustering():
    ndim, cell_names = 50, []
    infile = open("./plot/Schizo_Control/prediction/cell_names.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        cell_names.append(line[1])
    g = knn(np.load('./plot/Schizo_Control/prediction/cell_mCG_all_integrated_svd100.npy')[:, :ndim], n_neighbors=20)
    inter = g.dot(g.T)
    diag = inter.diagonal()
    jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
    adj = nx.from_numpy_matrix(np.multiply(g, jac)) #.toarray())
    knnjaccluster = {}
    for res in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10,15,20,25,50,100,200,500]:
        partition = community.best_partition(adj,resolution=res)
        label = np.array([k for k in partition.values()])
        knnjaccluster[res] = label
        nc = len(set(label))
        count = np.array([sum(label==i) for i in range(nc)])
        print(res, nc, count)

        infile = open("./plot/Schizo_Control/prediction/cell_mCG_all_integrated_svd50_p50_rs0.txt", "r")
        output = open("./plot/Schizo_Control/prediction_cluster_"+str(res)+"_"+str(nc)+".txt", "w")
        for idx,line in enumerate(infile):
            line = line.strip("\n")
            output.write(cell_names[idx] + "\t" + line + "\t" + "cell" +str(knnjaccluster[res][idx]) + "\n")
        output.close()
    np.save('./plot/Schizo_Control/prediction/cell_mCG_all_integrated_svd50_knn20_louvain.npy', knnjaccluster)


def RNAseq_embedding():
    rate_bin = np.load('./outputs/rnaseq/prediction/cell_expression.npy')
    rate_bin = np.nan_to_num(rate_bin, nan=0, posinf=0)

    bin_all = []
    infile = open("./outputs/rnaseq/prediction/gene_names.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        bin_all.append(line)
    bin_all = np.array(bin_all)

    binlist, data = [], []
    s = np.sum(rate_bin > 0, axis=0)
    binfilter = s >= 0.2 * len(rate_bin)
    print(len(rate_bin), np.sum(binfilter))
    rateb = rate_bin[:, binfilter]
    for i in range(np.sum(binfilter)):
        rateb[rateb[:,i]==0, i] = np.mean(rateb[rateb[:,i]>0, i])
    data.append(rateb)
    binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

    integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
    rateb_reduce = np.concatenate(integrated)
    n_components, ndim = 2, 50
    tsne = TSNE(n_components)
    y = tsne.fit_transform(rateb_reduce[:, :ndim])

    np.save('./datasets/matrix/expression/cell_RNA_all_prediction_svd100.npy', rateb_reduce)
    np.savetxt('./datasets/matrix/expression/cell_RNA_all_prediction_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')


def RNA_clustering():
    ndim, cell_names = 50, []
    infile = open("./outputs/rnaseq/prediction/cell_names.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        cell_names.append(line[0])
    g = knn(np.load('./datasets/matrix/expression/cell_RNA_all_prediction_svd100.npy')[:, :ndim], n_neighbors=20)
    inter = g.dot(g.T)
    diag = inter.diagonal()
    jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
    adj = nx.from_numpy_matrix(np.multiply(g, jac)) #.toarray())
    knnjaccluster = {}
    for res in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.6,2.0,2.5,3.0,4.0,5.0,10,15,20,25,30,35,40,45,50]:
        partition = community.best_partition(adj,resolution=res)
        label = np.array([k for k in partition.values()])
        knnjaccluster[res] = label
        nc = len(set(label))
        count = np.array([sum(label==i) for i in range(nc)])
        print(res, nc, count)
        infile = open("./datasets/matrix/expression/cell_RNA_all_prediction_svd50_p50_rs0.txt", "r")
        output = open("./plot/RNAseq/prediction_"+str(res)+"_"+str(nc)+".txt", "w")
        for idx,line in enumerate(infile):
            line = line.strip("\n")
            output.write(cell_names[idx] + "\t" + line + "\t" + "cell" +str(knnjaccluster[res][idx]) + "\n")
        output.close()
    np.save('./datasets/matrix/expression/cell_RNA_all_prediction_svd50_knn20_louvain.npy', knnjaccluster)
    output.close()


# embedding by mCG
def mCG_embedding():
    bin_all = "Br1416"
    infile = open("./datasets/matrix/Schizo_Control/single_eval/cell_names.txt", "r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        cell_name, sample_name, type_name = line[0], line[1], line[2]

    rate_bin = np.load('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_rate.100kbin.mCG.npy')
    read_bin = np.load('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_read.100kbin.mCG.npy')
    cell_meta = np.load('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_meta.genome.mCG.npy')    
    infile = open("./datasets/matrix/bin_all.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        bin_all.append(line)
    bin_all = np.array(bin_all)
    rate_tmp = rate_bin
    bin_variances = abs(np.mean(rate_tmp[:, :],axis=0)-np.mean(rate_tmp[:, :], axis=0))
    
    binlist, data, read_threshold = [], [], 10
    s = np.sum(read_bin > 0, axis=0)
    binfilter = np.logical_and((np.std(rate_bin, axis=0)>0), bin_variances>0.02)
    binfilter = np.logical_and((s >= 0.5*len(read_bin)), binfilter)
    print(len(rate_bin), sum(binfilter))
    readb = read_bin[:, binfilter]
    rateb = rate_bin[:, binfilter]
    for i in range(np.sum(binfilter)):
        rateb[readb[:,i]<read_threshold, i] = np.mean(rateb[readb[:,i]>=read_threshold, i])
    rateb = np.divide((rateb[:,:]-np.mean(rateb[:,:],axis=0)), np.std(rateb[:,:],axis=0).astype(float))
    data.append(rateb)
    binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])
    
    integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
    rateb_reduce = np.concatenate(integrated)
    n_components, ndim = 2, 100
    tsne = TSNE(n_components)
    y = tsne.fit_transform(rateb_reduce[:, :ndim])

    np.save('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_mCG_all_integrated_svd100.npy', rateb_reduce)
    np.savetxt('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_mCG_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')


# clustering by mC
def mCG_clustering():
    Phenotypes = {"Br1416":"Schizo","Br1446":"Control","Br1682":"Schizo","Br1092":"Control",
            "Br2288":"Control","Br1425":"Schizo","Br1245":"Schizo","Br1378":"Control"}
    ndim, bin_all, cell_names = 100, [], []
    infile = open("./datasets/matrix/Schizo_Control/single_eval/cell_names.txt", "r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        cell_name, sample, type_name = line[0], line[1], line[2]
        cell_names.append([cell_name, sample, Phenotypes[sample], type_name])
    g = knn(np.load('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_mCG_all_integrated_svd100.npy')[:, :ndim], n_neighbors=20)
    inter = g.dot(g.T)
    diag = inter.diagonal()
    jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
    adj = nx.from_numpy_matrix(np.multiply(g, jac)) #.toarray())
    knnjaccluster = {}
    for res in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.4,1.6,1.8,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10,15,20,25,30,35,40,45,50,55,60,65,70,80,90,100]:
        partition = community.best_partition(adj,resolution=res)
        label = np.array([k for k in partition.values()])
        knnjaccluster[res] = label
        nc = len(set(label))
        count = np.array([sum(label==i) for i in range(nc)])
        print(res, nc, count)
        infile = open("./datasets/matrix/Schizo_Control/"+sample_name+"/cell_mCG_all_integrated_svd50_p50_rs0.txt", "r")
        output = open("./plot/Schizo_Control/"+sample_name+"/raw_cluster_"+str(res)+"_"+str(nc)+".txt", "w")
        for idx,line in enumerate(infile):
            line = line.strip("\n")
            cell_name, sample, phenotype, type_name = cell_names[idx][0],cell_names[idx][1],cell_names[idx][2],cell_names[idx][3]
            output.write(cell_name+"\t"+line+"\t"+"cell"+str(knnjaccluster[res][idx])+"\t"+phenotype+"\t"+type_name+"\n")
        output.close()
    np.save('./datasets/matrix/Schizo_Control/'+sample_name+'/cell_mCG_all_integrated_svd50_knn20_louvain.npy', knnjaccluster)
    output.close()


# clustering neurons by mCH
def mCH_embedding():
    partion = "total_cov"
    knnjaccluster = np.load('./datasets/matrix/snm3cseq/'+partion+'/cell_mCG_all_integrated_svd50_knn20_louvain.npy', allow_pickle=True).item()
    label = knnjaccluster[1.6]
    neufilter = np.array([x in [0,1,3,4,6,8,12,13,14] for x in label])
    rate_bin = np.load('./datasets/matrix/snm3cseq/'+partion+'/cell_rate.100kbin.mCH.npy')[neufilter]
    read_bin = np.load('./datasets/matrix/snm3cseq/'+partion+'/cell_read.100kbin.mCH.npy')[neufilter]
    cell_meta = np.load('./datasets/matrix/snm3cseq/'+partion+'/cell_meta.genome.mCH.npy')
 
    infile = open("./Methyl/Matrix/Snm3c_CG/cell_names.txt","r")
    cell_names, batch, indiv, bin_all = [], [], [], []
    for line in infile:
        line = line.strip("\n").split("\t")
        cell_names.append(line[1])
        line = line[1].split("_")
        batch.append(line[2])
        indiv.append(line[5])
    infile = open("./datasets/matrix/bin_all.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        bin_all.append(line)
    batch, indiv, bin_all = np.array(batch), np.array(indiv), np.array(bin_all)

    batlist, indlist, binlist, data = list(set(batch.tolist())), list(set(indiv.tolist())), [], []
    neubatch = batch[neufilter]
    neuindiv = indiv[neufilter]
    neumeta = cell_meta[neufilter]
    for bat in batlist:
        for ind in indlist:
            cell = np.logical_and(neubatch==bat, neuindiv==ind)
            if np.sum(cell) <= 0: continue
            rate_tmp = rate_bin[cell]
            read_tmp = read_bin[cell]
            s = np.sum(read_tmp>100, axis=0)
            binfilter = np.logical_and((s>=0.99*len(read_tmp)), bin_all[:,0]!='chrX')
            print(sum(cell), sum(binfilter))
            rateb = np.divide(rate_tmp[:, binfilter].T, np.max(neumeta[cell]).astype(float)).T
            readb = read_tmp[:, binfilter]
            for i in range(np.sum(binfilter)):
                rateb[readb[:,i]<100, i] = np.mean(rateb[readb[:,i]>=100, i])
            data.append(rateb)
            binlist.append(['-'.join(x.tolist()) for x in bin_all[binfilter]])

    integrated, corrected, genes = scanorama.correct(data, binlist, return_dimred=True, hvg=None)
    rateb_reduce = np.concatenate(integrated)
    n_components, ndim  = 2, 50
    tsne = TSNE(n_components)
    y = tsne.fit_transform(rateb_reduce[:, :ndim])

    np.save('./datasets/matrix/snm3cseq/'+partion+'/neu_mCH_all_integrated_svd100.npy', rateb_reduce)
    np.savetxt('./datasets/matrix/snm3cseq/'+partion+'/neu_mCH_all_integrated_svd50_p50_rs0.txt', y, delimiter='\t', fmt='%s')


def mCH_clustering():
    ndim, partion, cell_names = 50, "total_cov", []
    infile = open("./Methyl/Matrix/Snm3c_CG/cell_names.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        cell_names.append(line[1])
    rateb_reduce = np.load('./datasets/matrix/snm3cseq/'+partion+'/neu_mCH_all_integrated_svd100.npy')
    g = knn(rateb_reduce[:, :ndim], n_neighbors=20)
    inter = g.dot(g.T)
    diag = inter.diagonal()
    jac = inter.astype(float) / (diag[None, :] + diag[:, None] - inter)
    adj = nx.from_numpy_matrix(np.multiply(g, jac))
    knnjaccluster = {}
    for res in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]:
        partition = community.best_partition(adj,resolution=res)
        label = np.array([k for k in partition.values()])
        knnjaccluster[res] = label
        nc = len(set(label))
        count = np.array([sum(label==i) for i in range(nc)])
        print(res, count)
    np.save('./datasets/matrix/snm3cseq/'+partion+'/neu_mCH_all_integrated_svd50_knn20_jac_louvain.npy', knnjaccluster)

    label = np.load('./datasets/matrix/snm3cseq/'+partion+'/neu_mCH_all_integrated_svd50_knn20_jac_louvain.npy', allow_pickle=True).item()[1.6]
    nc = len(set(label))
    selc = [[0,1,4,14],[10],[3,5,7],[6,16],[8,11,17],[12,18],[9,19],[2,13,15]]
    neuleg = ['' for i in range(nc)]
    leg = ['L2/3', 'L4', 'L5', 'L6', 'Ndnf', 'Vip', 'Pvalb', 'Sst']
    for xx,yy in zip(selc,leg):
        for i,c in enumerate(xx):
            neuleg[c] = yy+'-'+str(i+1)
    neulabel = np.array([neuleg[x] for x in label])

    label = np.load('./datasets/matrix/snm3cseq/'+partion+'/cell_mCG_all_integrated_svd50_knn20_louvain.npy', allow_pickle=True).item()[1.6]
    nc = len(set(label))
    selc = [[0,1,3,4,6,8,12,13,14],[5],[2,9],[7],[11],[10],[15,16,17]]
    allleg = ['' for i in range(nc)]
    leg = ['Neuron', 'Astro', 'ODC', 'OPC', 'MG', 'MP', 'Endo']
    for xx,yy in zip(selc,leg):
        for i,c in enumerate(xx):
            allleg[c] = yy+'-'+str(i+1)
    alllabel = np.array([allleg[x] for x in label])

    neumerge = np.array([x.split('-')[0] for x in neulabel])
    allmerge = np.array([x.split('-')[0] for x in alllabel])

    cluster = allmerge.copy()
    cluster[allmerge=='Neuron'] = neumerge.copy()
    subcluster = alllabel.copy()
    subcluster[allmerge=='Neuron'] = neulabel.copy()

    infile = open("./datasets/matrix/snm3cseq/"+partion+"/cell_mCG_all_integrated_svd50_p50_rs0.txt", "r")
    output = open("./plot/Snm3c_CG/total_cluster.txt", "w")
    for idx,line in enumerate(infile):
        line = line.strip("\n")
        output.write(cell_names[idx] + "\t" + line + "\t" + str(allmerge[idx]) + "\n")

    infile = open("./datasets/matrix/snm3cseq/"+partion+"/cell_mCH_all_integrated_svd50_p50_rs0.txt", "r")
    output = open("./plot/Snm3c_CH/total_cluster.txt", "w")
    for idx,line in enumerate(infile):
        line = line.strip("\n")
        output.write(cell_names[idx] + "\t" + line + "\t" + str(allmerge[idx]) + "\n")
    output.close()


def read_cell_type():
    cell_types, methylation_data = {}, {} 
    infile = open("./plot/Snm3c_CG/prediction/total_cluster_0.1_81.txt","r")
    for line in infile:
        line = line.strip("\n").split("\t")
        cell_name, cell_type = line[0], line[3]
        variation_types[cell_name] = cell_type
    for cell_name in variation_types.keys():
        reference_type = reference_types[cell_name]
        variation_type = variation_types[cell_name]
        pair_name = variation_type + "&" + reference_type
        if pair_name not in counts.keys(): counts[pair_name] = 1
        else: counts[pair_name] = counts[pair_name] + 1
    for pair_name in counts.keys():
        if counts[pair_name] < 10: continue
        print(pair_name + "\t" + str(counts[pair_name]))


def Adjusted_rand_index():
    type_index = {"CGEInh":0, "MGEInh":1, "SupExc":2, "DeepExc":3, "Glia":4}
    reference_types, variation_types, cluster_types, prediction_types = [], [], [], []
    infile = open("./plot/Snmc_CG/total_cluster_1.5_16.txt","r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        type_name, cell_name = line[3], line[0]
        reference_types.append(int(type_name[4:]))
    
    infile = open("./plot/Snmc_CG/one_percent_cluster_1.6_14.txt","r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        type_name, cell_name = line[3], line[0]
        variation_types.append(int(type_name[4:]))
    
    infile = open("./plot/Snmc_CG/cluster/one_percent_cluster_50_17.txt","r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        type_name, cell_name = line[3], line[0]
        cluster_types.append(int(type_name[4:]))
    
    infile = open("./plot/Snmc_CG/prediction/one_percent_cluster_1.0_16.txt","r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        type_name, cell_name = line[3], line[0]
        prediction_types.append(int(type_name[4:]))

    Ari = adjusted_rand_score(reference_types, variation_types)
    print("Down sampling Adjusted Rand Index: " + str(Ari))
    
    Ari = adjusted_rand_score(reference_types, cluster_types)
    print("cluster Adjusted Rand Index: " + str(Ari))
    
    Ari = adjusted_rand_score(reference_types, prediction_types)
    print("Prediction Adjusted Rand Index: " + str(Ari))


if __name__ == "__main__":
    #split_embedding()
    #split_clustering()
    #prediction_embedding()
    #prediction_clustering()
    mCG_embedding()
    mCG_clustering()
    #read_cell_type()
    #Adjusted_rand_index()
