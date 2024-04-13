#!/usr/local/bin/python

#############################################################
import sys
import math
import numpy as np

#global parameter
letter_length = 1
gmax_word_length = 12
word_len_list = range(1,13)
DELIMITER = "\ "

class Tokenizer:
    def __init__(self):
        self.word_dict = {}
        self.word_id = {}

    #segment function
    def tokenize(self, sequence, window):
        sequence = sequence.strip()

        #step 1, construct segment graph and find the best path

        # s_left recod the sum of probability, for example s_left[5] record the sum
        #probability in postion 5
        s_left = []

        # s_left_set record the left segment point, for example, s_left_seg[5] record
        # the left segment point of postion 5 in sequence
        s_left_seg = []
        s_left.append(0)
        s_left_seg.append(0)

        for pos in range(1,len(sequence) + letter_length,letter_length):

            if pos < gmax_word_length:
                max_word_length = pos
            else:
                max_word_length = gmax_word_length

            max_prob = -1*sys.maxsize
            left_seg = 0
            for length  in range(letter_length,max_word_length+letter_length,letter_length):
                word = sequence[pos-length:pos]
                if word in self.word_dict.keys():
                    prob = self.word_dict[word] + s_left[(pos-length)//letter_length]
                    if prob > max_prob:
                        max_prob = prob
                        left_seg = pos-length
            s_left.append(max_prob)
            s_left_seg.append(left_seg)

        # step 2, get segment point
        seg_pos=[]
        seg_pos.append(len(sequence))
        pos = s_left_seg[-1]
        seg_pos.append(pos)
        while True:
            if pos == 0:
                break
            pos = s_left_seg[pos//letter_length]
            seg_pos.append(pos)
        seg_pos.reverse()

        # step 3, create segmented words list
        seg_sequence=""
        word_list = np.zeros(window)
        for pos in range(0,len(seg_pos)-1):
            left = seg_pos[pos]
            right = seg_pos[pos + 1]
            word = sequence[left:right]
            if pos >= window: break
            word_list[pos] = self.word_id[word]
            
        return word_list

    #initial dict
    def initial_dict(self,filename):
        dict_file = open(filename,'r')
        for idx,line in enumerate(dict_file):
            sequence = line.strip()
            key = sequence.split('\t')[0]
            value = float(sequence.split('\t')[1])
            if len(key) in word_len_list:
                self.word_dict[key] = value
                self.word_id[key] = idx+1

#test
if __name__=='__main__':
    myseg = DNASegment()
    myseg.initial_dict("./data/vocab.txt")
    infile = open("/dcs04/lieber/statsgen/jiyunzhou/ELECTRA_PLAC/datasets/corpus/contact/promoter_enhancer_0.txt","r")
    for line in infile:
        line = line.strip("\n").split()
        promoter,enhancer = line[0],line[1]
        promoter_word_list = myseg.mp_seg(promoter)
        enhancer_word_list = myseg.mp_seg(enhancer)
        print(enhancer_word_list)
        print("promoter: %d\tenhancer: %d" % (len(promoter_word_list),len(enhancer_word_list)))
