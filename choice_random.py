import numpy as np













def INDEX(x,num_clss,samples):

    idx_per_clss=samples//num_clss

    sampled_idx=[]

    for clss_vlu in range(num_clss):

        clss_idx=np.where(x==clss_vlu)[0]
        sampled_clss_idx=np.random.choice(clss_idx,size=idx_per_clss,replace=False)
        sampled_idx.extend(sampled_clss_idx)

    sampled_idx=np.array(sampled_idx)
    np.random.shuffle(sampled_idx)
    return sampled_idx


    
# Example usage
if __name__ == "__main__":
    #print('...............',(int(np.ceil(1000/128)))*16)


    vector_r=np.random.choice([0,1,2,3],size=287)
    total_indx_pick=40

    num_clss=4


    output=INDEX(vector_r,num_clss,total_indx_pick )



    


    print("sampled Index",output)