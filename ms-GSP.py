
# * cs583 Datamining and Text Mining
# * Programming Assignment 1
# * Team member:
# * Huiyang Zhao 655490960
# * Yimai Zhao   675695155
import copy
from functools import cmp_to_key
import numpy as np
import sys

# global itemMap: a hashmap (item, count)
# global misMap: a hashmap (item, mis(item))
itemMap = {}
misMap = {}
sequenceData = []
rest = 0.01

# msfile = './test-data/small-data-1/para1-1.txt'
# msfile = './test-data/small-data-1/para1-2.txt'
# msfile = './test-data/large-data-2/para2-1.txt'
msfile = './test-data/large-data-2/para2-2.txt'

# sequencefile = './test-data/small-data-1/data-1.txt'
sequencefile = './test-data/large-data-2/data2.txt'

# outputfile = './output-data/output_small_1-1.txt'
# outputfile = './output-data/output_small_1-2.txt'
# outputfile = './output-data/output_large_2-1.txt'
outputfile = './output-data/output_large_2-2.txt'

def ms_gsp():
    F_results = []

    # input MS: mis and sdc
    # output M: sorted item list.
    def sort():
        global misMap
        global sdc
        global rest

        pre_misMap = {}
        # misFile = open('./test_ms.txt', 'r')
        misFile = open(msfile, 'r')
        # misFile = open(msfile[index], 'r')

        while True:
            # read file line by line and record mis for each item.
            line = misFile.readline().strip('\n')
            if 'SDC' in line:
                sdc = float(line[7:])
                continue
            if 'rest' in line:
                rest = float(line[13:])
                continue
            if not line:
                break
            line_after = line.strip('MIS(').replace(') = ', ',').split(',')
            pre_misMap[int(line_after[0])] = float(line_after[1])

        misFile.close()

        # with open('./test_sequences.txt', 'r') as f:
        with open(sequencefile, 'r') as f:
        # with open(sequencefile[index], 'r') as f:
            while True:
                # read file line by line and record support in sequence.
                line = f.readline()
                # count number of sequence```
                if not line:
                    break
                if line == '\n':
                    continue
                line_mid = line.replace('<', '').replace('>', '').replace('\n', '').replace('}{', '.'). \
                    replace('{', '').replace('}', '').split('.')
                line_after = [itemList.split(',') for itemList in line_mid]

                line_final = []
                for itemList in line_after:
                    # print(itemList)
                    line_final.append([int(item) for item in itemList])
                # print(line_final)
                for itemSet in line_final:
                    for item in itemSet:
                        if item not in pre_misMap.keys():
                            pre_misMap[item] = rest

        keys = list(pre_misMap.keys())
        values = list(pre_misMap.values())

        sorted_value_index = np.argsort(values)
        misMap = {keys[i]: values[i] for i in sorted_value_index}

    # output: itemset number n
    # output: L candidate 1-sequence
    def init_pass():  # return a set of items I lager than its own MIS and their count
        # count_all_items = {}
        global n
        # global misMap

        def compare(o1, o2):
            # return misMap_withoutRes[o1] - misMap_withoutRes[o2]
            return misMap[o1] - misMap[o2]

        # sequenceFile = open('./test_sequences.txt', 'r')
        sequenceFile = open(sequencefile, 'r')
        # sequenceFile = open(sequencefile[index], 'r')
        n = 0
        while True:
            # read file line by line and record support in sequence.
            line = sequenceFile.readline()
            # count number of sequence
            if not line:
                break
            if line == '\n':
                continue
            n += 1
            line_mid = line.replace('<', '').replace('>', '').replace('\n', '').replace('}{', '.'). \
                replace('{', '').replace('}', '').split('.')
            line_after = [itemList.split(',') for itemList in line_mid]

            line_final = []
            for itemList in line_after:
                # print(itemList)
                line_final.append(sorted([int(item) for item in itemList], key=cmp_to_key(compare)))
                # line_final.append([int(item) for item in itemList])
            # print(line_final)

            sequenceData.append(line_final)
            # print(line_final)

            # count number for every F-1 Sequence.
            line_for_count = [int(num) for num in line.replace('}{', ',').replace('<', '').replace('>', '').
                replace('{', '').replace('}', '').replace('\n', '').split(',')]
            itemSet = set()
            for item in line_for_count:
                itemSet.add(item)
            for item in itemSet:
                if item not in itemMap:
                    itemMap[item] = 1
                else:
                    itemMap[item] += 1

        sequenceFile.close()

        L = []
        first_item = 0
        first_item_index = 0
        first_item_mis = 0

        for i, (item, mis) in enumerate(misMap.items()):
            if item in itemMap and itemMap[item] / n > mis:
                first_item = item
                first_item_mis = mis
                L.append(item)
                break
            first_item_index += 1

        # for item, mis in misMap:
        for i, (item, mis) in enumerate(misMap.items()):
            if i <= first_item_index:
                # print('passed i: ' + str(i))
                continue
            if item in itemMap and itemMap[item] >= first_item_mis:
                # print('added i: ' + str(i))
                L.append(item)

        return L

    def contain(S, C):
        cont = 0
        # for s in S:
        flag = 0
        for e in S:
            if len(set(C[flag]) & set(e)) == len(C[flag]):
                flag += 1

            if flag == len(C):
                cont+=1
                flag = 0

        return cont

    def level2_candidate_gen_SPM(L):
        global misMap
        C2 = []
        itemsts_sorted = L
        for l in range(len(itemsts_sorted)):
            if itemMap[itemsts_sorted[l]] / n >= misMap[itemsts_sorted[l]]:
                C2.append([[itemsts_sorted[l]], [itemsts_sorted[l]]])
                for h in range(l + 1, len(itemsts_sorted)):
                    if itemMap[itemsts_sorted[h]] / n >= misMap[itemsts_sorted[l]] and abs(
                        itemMap[itemsts_sorted[l]] / n - itemMap[itemsts_sorted[h]] / n) <= sdc:
                        C2.append([[itemsts_sorted[l], itemsts_sorted[h]]])
                        C2.append([[itemsts_sorted[l]], [itemsts_sorted[h]]])
                        C2.append([[itemsts_sorted[h]], [itemsts_sorted[l]]])
        return C2

    def mscandidate_gen_SPM(F_k_minus1):
        C = []
        # print(F_k_minus1)
        for s1 in F_k_minus1:

            for s2 in F_k_minus1:
                if id(s1) == id(s2):
                    break

                s1_flat = flatten(s1)
                s2_flat = flatten(s2)
                
                Len_lastitem_separate = False
                Len_firstitem_separate = False

                if len(s2[-1]) == 1:
                    Len_lastitem_separate = True

                if len(s2[0]) == 1:
                    Len_firstitem_separate = True

                s1_len_2_size_2 = False

                s1_len_2_size_1 = False

                if len(s1) == 2 and len(s1[0]) == 1 and len(s1[1]) == 1:
                    s1_len_2_size_2 = True

                if len(s1) == 1 and len(s1[0]) == 2:
                    s1_len_2_size_1 = True

                mis_first_s2 = misMap[s2[0][0]]

                mis_last_s2 = misMap[s2[-1][-1]]

                mis_first_s1 = misMap[s1[0][0]]

                mis_last_s1 = misMap[s1[-1][-1]]

                minmis, index = min_mis(s1)
                
                if index == 0:
                    # print('the first one is s1', s1, s2)

                    if s1_flat[:1] + s1_flat[2:] == s2_flat[:-1] and mis_last_s2 > mis_last_s1:

                        if Len_lastitem_separate:  # half_verified
                            s2_last_seperate_tem = copy.deepcopy(s1)
                            s2_last_seperate_tem.append(s2[-1])
                            # print(s2_last_seperate_tem)
                            C.append(s2_last_seperate_tem)

                            if s1_len_2_size_2 and mis_last_s2 > mis_last_s1:  # half_verified
                                s1_len2size2_tem = copy.deepcopy(s1)
                                s1_len2size2_tem[-1].append(s2[-1][-1])
                                # print(s1_len2size2_tem)
                                C.append(s1_len2size2_tem)

                        elif s1_len_2_size_1 and mis_last_s2 > mis_last_s1 or len(s1_flat) > 2:  # half_verified
                            s1_len2size1_tem = copy.deepcopy(s1)
                            s1_len2size1_tem[-1].append(s2[-1][-1])
                            # print(s1_len2size1_tem)
                            C.append(s1_len2size1_tem)

                elif index == len(s1_flat) - 1:
                    # print(1)
                    # print('the last one is s1', s1, s2)

                    if s1_flat[:-2] + [s1_flat[-1]] == s2_flat[1:]:
                        # print(s1,s2)

                        if Len_firstitem_separate:  # verified!
                            s2_first_seperate_tem = [s2[0]]
                            s2_first_seperate_tem.extend(s1)
                            # print(s2_first_seperate_tem)
                            C.append(s2_first_seperate_tem)

                            if s1_len_2_size_2 and mis_first_s2 < mis_first_s1:  # verified!
                                # print(s1, s2)

                                s1_len2size2_tem = copy.deepcopy(s2)
                                s1_len2size2_tem[0].append(s1[0][0])
                                # print(s1, s2)
                                # print(s1_len2size2_tem)
                                C.append(s1_len2size2_tem)
                                # print(s1, s2)

                        elif s1_len_2_size_1 and mis_first_s2 < mis_first_s2 or len(s1_flat) > 2:  # half_verified!
                            s2_len2size1_tem = copy.deepcopy(s1)
                            tem_tem = [s2[0][0]]
                            tem_tem.extend(s1[0])
                            s2_len2size1_tem[0] = tem_tem
                            # print(s2_len2size1_tem)
                            C.append(s2_len2size1_tem)

                else:
                    # print(s1,s2)
                    # print(1)

                    if s1_flat[1:] == s2_flat[:-1]:
                        # print('regular', s1, s2)
                        # print(s1, s2)
                        if Len_lastitem_separate:
                            tem1 = copy.deepcopy(s1)
                            tem1.append([s2[-1][-1]])
                        else:
                            tem1 = copy.deepcopy(s1)
                            tem1[-1].append(s2[-1][-1])
                        # print(tem1)
                        C.append(tem1)

                    # return a list of candidate such as [([1,2,3]) , ([4],[4,5],[9)] when s2 = 3
        # print(C)
        C_after_prune = prune(C, F_k_minus1)
        # print(C_after_prune)
        return C_after_prune

    def min_mis(c):
        # print(c)
        # since none of them cant be above 1

        minimum = [2, -1]

        index = 0
        # print(c)
        for i in c:
            # print(c)
            # print(i)
            for j in i:
                # print(c)
                # print(j)
                if misMap[j] < minimum[0]:
                    minimum[0] = misMap[j]
                    minimum[1] = index
                elif misMap[j] == minimum[0]:
                    minimum[1] = -1
                index += 1

        return minimum


    def output(F_results):
        original_stdout = sys.stdout
        # with open('./output.txt', 'w') as outputFile:
        with open(outputfile, 'w') as outputFile:
        # with open(outputfile[index], 'w') as outputFile:
            sys.stdout = outputFile
            for k,i in enumerate (F_results):
                print('************************************** ')

                print(k+1,'-sequences: \n')
                count = len(i)
                for j in i:
                    tem = str(j)
                    tem = '<' + tem[1:-1] + '>'
                    tem = tem.replace('[', '{').replace('], ', '}').replace(' ', '').replace(']', '}')
                    print(tem)
                print('\nThe count is:', count)
            sys.stdout = original_stdout

        # print as the format required

    def flatten(s):
        tem = []
        for i in s:
            for j in i:
                tem.append(j)
        return tem

    def prune(big_c, f_minus1):
        # print(big_c)
        def scan(cand, f_minus1):
            flag = False
            for f in f_minus1:
                if f == cand:
                    flag = True
                    break
            return flag


        F = []

        for c in big_c:
            index = min_mis(c)
            index = index[1]
            # print(c)
            # print('index',index)
            count = 0
            flag1 = True
            for i in range(len(c)):
                if count == index:
                    continue
                if len(c[i]) == 1:
                    cand = c[:i] + c[i + 1:]
                    # print('cand',cand)
                    # print('result',scan(cand, f_minus1))
                    if scan(cand, f_minus1):
                        count += 1
                        continue
                    else:
                        flag1 = False
                        break
                else:
                    for j in range(len(c[i])):
                        cand = copy.deepcopy(c)
                        tem2 = copy.deepcopy(c[i])
                        del tem2[j]
                        cand[i] = tem2
                        # print('cand', cand)
                        # print('result', scan(cand, f_minus1))
                        if scan(cand, f_minus1):
                            count += 1
                            continue
                        else:
                            flag1 = False
                            break
            if flag1:
                F.append(c)
                # print(F)
        return F

    sort()
    L = init_pass()

    F1 = []
    for l in L:
        if itemMap[l] / n > misMap[l]:
            F1.append([[l]]) # for the format to match 3d
    F_results.append(F1)

    k = 2

    F_minus1 = F1
    Fk = []
    while 1:

        # F_minus1 should be same format as candidate like [([1,2,3]) , ([4],[4,5],[9)]

        if k == 2:
            CK = level2_candidate_gen_SPM(L)

        else:
            CK = mscandidate_gen_SPM(F_minus1)
        k += 1
        if len(CK) == 0:
            break  # in case we don't have any candidates
        c_count = {}
        for s in sequenceData:
            for c in CK:
                if contain(s, c) > 0:
                    # ???how to scan whole sequence with F_minus1???
                    if str(c) in c_count:
                        c_count[str(c)] += 1
                    else:
                        c_count[str(c)] = 1
        for c in CK:
            # print(c)
            if str(c) in c_count and c_count[str(c)] / n >= min_mis(c)[0]:
                Fk.append(c)
        if len(Fk) == 0:
            break
        F_minus1 = Fk
        Fk = []
        F_results.append(F_minus1)

    output(F_results)

ms_gsp()