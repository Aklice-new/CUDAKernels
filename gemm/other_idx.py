

for tid in range(0, 32):
    # int load_smem_a_m = tid / 2;
    # int load_smem_a_k = (tid & 1) << 2;
    # int load_smem_b_k = tid / 32;
    # int load_smem_b_n = (tid & 31) << 2;
    print("smem_a_m : ", tid // 2, end=' ')
    print("smem_a_k : ", (tid & 1) << 2, end=' ')
    print("smem_b_k : ", tid // 32, end=' ')
    print("smem_b_n : ", (tid & 31) << 2)


