

for tid in range(0, 32):
    # int load_smem_a_m = tid / (BK / 4); //  BK / TM = 2, 加载一行需要两个线程
    # int load_smem_a_k = (tid % (BK / 4)) * 4;
    # int load_smem_b_k = tid / (BN / 4);
    # int load_smem_b_n = (tid % (BN / 4)) * 4;
    print("smem_a_m : ", tid // (8 // 4), end=' ')
    print("smem_a_k : ", (tid % (8 // 4)) * 4, end=' ')
    print("smem_b_k : ", tid // (128 // 4), end=' ')
    print("smem_b_n : ", (tid % (128 // 4)) * 4)