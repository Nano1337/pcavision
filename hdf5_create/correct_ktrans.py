import shutil

train_set = True
i = 0
while i < 202:
    fnum = '%04d' % i
    fnum1 = '%04d' % (i+1)
    patient_id = 'ProstateX-' + fnum
    print(patient_id)
    if train_set:
        source = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\All_training_data\\ProstateX-' + fnum1 + '\\Ktrans'
        destination = 'C:\\Users\\haoli\\Downloads\\ProstateX_Data\\All_training_data\\ProstateX-' + fnum
        shutil.move(source, destination)
    # else:
        # shutil.copytree(sub_dirs_src1[i], main_path + '\\All_test_data\\' + patient_id)
        # shutil.copytree(sub_dirs_src2[i], main_path + '\\All_test_data\\' + patient_id + '\\Ktrans')
    i += 1