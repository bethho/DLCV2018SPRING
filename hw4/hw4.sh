wget 'https://www.dropbox.com/s/nwc1rzcqp2txnxd/vae_model.h5'
wget 'https://www.dropbox.com/s/1c5t8fx0fat24i2/generator_model_15000.h5'
wget 'https://www.dropbox.com/s/cbr2qjq36mpxu5n/gan_model_50000.h5'
python3 hw4_vae.py $1 $2
python3 hw4_gan.py $1 $2
python3 hw4_acgan.py $1 $2
