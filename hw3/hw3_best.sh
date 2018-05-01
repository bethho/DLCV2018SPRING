wget 'https://www.dropbox.com/s/twkb7ohm02ydxq4/U_net_VGG16_model_best.h5'
wget 'https://www.dropbox.com/s/mksamm1em4y6e60/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
python3 hw3_U_net_test.py $1 $2
