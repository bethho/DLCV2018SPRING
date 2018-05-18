wget 'https://www.dropbox.com/s/crd1pj72g125qb3/vae_model.h5'
wget 'https://www.dropbox.com/s/87n7u0q6rulojv4/generator_model_10000.h5'
wget 'https://www.dropbox.com/s/cbr2qjq36mpxu5n/gan_model_50000.h5'
python3 hw4_vae.py $1 $2
python3 hw4_gan.py $1 $2
python3 hw4_acgan.py $1 $2
