import numpy as np

file_01 = 'en_data/predicted.txt'
file_02 = 'en_output/predicted.txt'

hmm_01 = 'en_data/hmmemit.txt'
hmm_02 = 'en_output/hmmemit.txt'

hmm_03 = 'en_data/hmminit.txt'
hmm_04 = 'en_output/hmminit.txt'

hmm_05 = 'en_data/hmmtrans.txt'
hmm_06 = 'en_output/hmmtrans.txt'

def ifSame(txt_01, txt_02):
    list_01 = []
    list_02 = []

    with open(txt_01) as r_01:
        list_01 = r_01.readlines()

    with open(txt_02) as r_02:
        list_02 = r_02.readlines()    

    for i in range(len(list_02)):
        if list_01[i] != list_02[i]:
            print(f'Error! {i+1}', list_01[i], list_02[i])
        else:
            pass
            # print(f'{i} same!')
    # print(list_01[32959], list_02[32959])
    if len(list_01) != len(list_02):
        print("Different length!")
    else:
        print("length same!")
            
    # read_01 = np.loadtxt()

# ifSame(hmm_01, hmm_02)
# ifSame(hmm_03, hmm_04)
# ifSame(hmm_05, hmm_06)
ifSame(file_01, file_02)
