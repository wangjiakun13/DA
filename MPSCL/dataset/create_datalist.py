import os



def make_datalist(data_fd, gt_fd, data_list,data_gt_list):

    filename_all = os.listdir(data_fd)

    filename_all = [data_fd  + img_name + '\n' for img_name in filename_all if img_name.endswith('.npy')]

    gt_all = os.listdir(gt_fd)

    gt_all = [gt_fd + img_name + '\n' for img_name in gt_all if img_name.endswith('.npy')]


    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)

    with open(data_gt_list, 'w') as fp:
        fp.writelines(gt_all)




if __name__ == '__main__':

    #Plz change the path follow your setting
    data_fd      = '/home1/jkwang/dataset/MMWHS/data_np/val_ct/'
    gt_fd = '/home1/jkwang/dataset/MMWHS/data_np/gt_val_ct/'
    data_list    = '/home1/jkwang/dataset/MMWHS/data_np/data_list/val_ct.txt'
    data_gt_list = '/home1/jkwang/dataset/MMWHS/data_np/data_list/val_ct_gt.txt'
    make_datalist(data_fd, gt_fd, data_list, data_gt_list)

