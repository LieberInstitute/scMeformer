from collections import OrderedDict
from tokenization import Tokenizer
from pathlib import Path
import numpy as np
import random 
import sys
import json
import os

input_chrom = sys.argv[1]
window = 2001

chr_len = {"chr1":249250621,"chr2":243199373,"chr3":198022430,"chr4":191154276,"chr5":180915260,"chr6":171115067,\
           "chr7":159138663,"chr8":146364022,"chr9":141213431,"chr10":135534747,"chr11":135006516,"chr12":133851895,\
           "chr13":115169878,"chr14":107349540,"chr15":102531392,"chr16":90354753,"chr17":81195210,"chr18":78077248,\
           "chr19":59128983,"chr20":63025520,"chr21":48129895,"chr22":51304566}


def genome_CpG_position():
    genome = read_hg19()
    genome_cpg, idx = {}, 0
    start, end = 0, chr_len[input_chrom]
    content = genome[input_chrom][start:end]
    index = content.find("CG",0)
    while index >= 0:
        pos = index + start + 1
        genome_cpg[pos] = idx
        idx = idx + 1
        index = content.find("CG",index+1)
    np.save("./datasets/position/"+input_chrom, genome_cpg, allow_pickle=True)
    return genome_cpg


def label_sequence(sequence, MAX_SEQ_LEN):
    nucleotide_ind = {"N":0, "A":1, "T":2, "C":3, "G":4}
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(sequence):
        X[i] = nucleotide_ind[ch]
    return X


def read_chrom(chrom):
    chrom_data, cell_names, cell_types, cluster_data,  = {}, [], {}, {}
    infile = open("./plot/Schizo_Control/Br1092/single_cluster_0.2_28.txt","r")
    for idx,line in enumerate(infile):
        line = line.strip("\n").split()
        type_name, cell_name = line[3], line[0]
        cell_names.append(cell_name)
        cell_types[cell_name] = type_name

    for idx, cell_name in enumerate(cell_names):
        num_sample = 0
        infile = open("./Methyl/Methyl/LUO_SCZ_CG/"+chrom+"/"+cell_name+".tsv","r")
        line = infile.readline()
        for line in infile:
            line = line.strip("\n").split()
            pos, strand, mc_count, total_count = int(line[1]), line[2], float(line[4]), float(line[5])
            if random.random() > 0.01: continue
            
            num_sample = num_sample + 1
            if strand == "-": pos, strand = pos - 1, "+"
            if mc_count == 0: methyl_level = 0
            elif mc_count == total_count: methyl_level = 1
            else: continue
          
            type_name = cell_types[cell_name]
            if type_name not in cluster_data.keys(): cluster_data[type_name] = {}
            if pos not in cluster_data[type_name].keys():
                cluster_data[type_name][pos] = [chrom, pos, strand, mc_count, total_count]
            else:
                cluster_data[type_name][pos][3] = cluster_data[type_name][pos][3] + mc_count
                cluster_data[type_name][pos][4] = cluster_data[type_name][pos][4] + total_count

            type_name = "Brain"
            if type_name not in cluster_data.keys(): cluster_data[type_name] = {}
            if pos not in cluster_data[type_name].keys():
                cluster_data[type_name][pos] = [chrom, pos, strand, mc_count, total_count]
            else:
                cluster_data[type_name][pos][3] = cluster_data[type_name][pos][3] + mc_count
                cluster_data[type_name][pos][4] = cluster_data[type_name][pos][4] + total_count
 
            key = chrom + "_" + str(pos) + "_" + strand
            try: chrom_data[key][methyl_level].append(idx)
            except:
                chrom_data[key] = list(range(2))
                chrom_data[key][0], chrom_data[key][1] = [], []
                chrom_data[key][methyl_level].append(idx)
        print(str(idx) + "\t" + cell_name)
        print(str(len(chrom_data)) + "\t" + str(num_sample))
    return chrom_data, cluster_data

        
def Get_cell_chrom(chrom_data, chrom):
    methylation_data = []
    sample_num,length = 0,int(window/2)
    print("processing " + chrom)
    output = open("./datasets/Schizo_Control/Br1092/task_number/"+chrom+".txt","w")
    for idx, cpg_key in enumerate(chrom_data.keys()): 
        output.write(str(idx) + "\t" + str(len(chrom_data[cpg_key][0])+len(chrom_data[cpg_key][1])) + "\n")
        chrom, pos, strand = cpg_key.split("_")[0], int(cpg_key.split("_")[1]), cpg_key.split("_")[2]
        methyl_level = chrom_data[cpg_key]
        item = {"chrom": chrom, "pos": pos, "strand": strand, "high_methyl": methyl_level[1], "low_methyl": methyl_level[0]}
        methylation_data.append(item)
        sample_num = sample_num + 1 
    print("the number of samples: %d" % sample_num)
    output.close()
    json.dump(methylation_data, open("./datasets/Schizo_Control/Br1092/" + chrom + ".json","w"))
    return methylation_data


def Get_cluster_chrom(cluster_data, chrom):
    genome_cpg = np.load("./datasets/position/"+input_chrom+".npy",allow_pickle=True).item()
    num_cpg, num_feature = len(genome_cpg), len(cluster_data.keys())-1
    print("The number of cpg is " + str(num_cpg))
    cluster_feature = np.zeros((num_cpg, num_feature))

    for pos in cluster_data["Brain"].keys():
        index = genome_cpg[pos]
        methyl_level = cluster_data["Brain"][pos][3] / cluster_data["Brain"][pos][4]
        cluster_feature[index,:] = methyl_level

    del cluster_data["Brain"]
    for idx,cell_type in enumerate(cluster_data.keys()):
        print(cell_type)
        for pos in cluster_data[cell_type].keys():
            index = genome_cpg[pos]
            methyl_level = cluster_data[cell_type][pos][3] / cluster_data[cell_type][pos][4]
            cluster_feature[index, idx] = methyl_level
        print("The number of cpg is " + str(len(cluster_data)))
        print(cluster_feature[:, idx])
    np.save("./datasets/Schizo_Control/Br1092/feature_data/"+chrom, cluster_feature, allow_pickle=True)
    return cluster_feature 


def Get_genome_CpG(chrom):
    methylation_data = []
    genome_cpg = np.load("./datasets/position/"+input_chrom+".npy",allow_pickle=True).item()
    sample_num,length = 0,int(window/2)
    print("processing " + chrom)
    for idx, cpg_pos in enumerate(genome_cpg.keys()):
        pos, strand = cpg_pos, "+"
        item = {"chrom": chrom, "pos": pos, "strand": strand}
        methylation_data.append(item)
        sample_num = sample_num + 1
    print("the number of samples: %d" % sample_num)
    json.dump(methylation_data, open("./datasets/genome_cpg/" + chrom + ".json","w"))
    return methylation_data


def read_hg19():
    genome, size = {}, {}
    chromo, current_chr = "", ""
    DNA_file = open("./datasets/hg19.fa")
    for line in DNA_file:
        line = line.strip("\t\r\n")
        if ">chr" in line:
            if current_chr == "":
                line = line.split()
                current_chr = line[0][1:]
            else:
                genome[current_chr],size[current_chr] = chromo,len(chromo)
                chromo, line = "", line.split()
                current_chr = line[0][1:]
        elif ">" in line:
            genome[current_chr], size[current_chr] = chromo, len(chromo)
            break
        else: chromo = chromo + line
    for i in range(1,23):
        print("the length of chr %d is %d " % (i,size["chr"+str(i)]))
    return genome


if __name__=="__main__":
    #Get_genome_CpG(input_chrom)
    chrom_data, cluster_data = read_chrom(input_chrom)
    Get_cell_chrom(chrom_data, input_chrom)
    Get_cluster_chrom(cluster_data, input_chrom) 
