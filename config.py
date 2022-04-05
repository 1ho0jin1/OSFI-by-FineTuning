data_config = {
    "CASIA": {
        "G_data_dir": "/home/yoon/datasets/face/CASIA_clean/cropped_images",  # CASIA-WebFace/cropped_images directory
        "K_data_dir": "/home/yoon/datasets/face/CASIA_clean/cropped_images",  # CASIA-WebFace/cropped_images directory
        "U_data_dir": "/home/yoon/datasets/face/CASIA_clean/cropped_images",  # CASIA-WebFace/cropped_images directory
        "known_pkl": "/home/yoon/datasets/face/CASIA_clean/CASIA_known_list.pkl",  # CASIA_known_list.pkl directory
        "unknown_pkl": "/home/yoon/datasets/face/CASIA_clean/CASIA_unknown_list.pkl",  #  CASIA_unknown_list.pkl directory
    },
    "IJBC":{
        "G_data_dir" : "/home/yoon/datasets/face/IJB/IJB_C_Cropped/images_organized",  # IJBC/images_organized directory
        "K_data_dir" : "/home/yoon/datasets/face/IJB/IJB_C_Cropped/frames_organized",  # IJBC/frames_organized directory
        "U_data_dir" : "/home/yoon/datasets/face/IJB/IJB_C_Cropped/frames_organized",  # IJBC/frames_organized directory
        "known_pkl" : "/home/yoon/datasets/face/IJB/IJB_C_Cropped/IJBC_known_list.pkl",  # IJBC_known_list.pkl directory
        "unknown_pkl" : "/home/yoon/datasets/face/IJB/IJB_C_Cropped/IJBC_unknown_list.pkl",  # IJBC_unknown_list.pkl directory
    },
}

encoder_config = {  # pretrained weight directory
    "VGG19": "/data/New_Projects/OSFI/model/VGG19_CosFace.chkpt",
    "Res50": "/data/New_Projects/OSFI/model/ResIR50_CosFace.chkpt",
}
