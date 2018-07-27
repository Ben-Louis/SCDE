import sys
import os
sys.path.append(os.path.abspath(__file__).rsplit('/',1)[0])

from .Encoder import Encoder_asmb as Encoder_pix2pix
from .Decoder import Decoder_asmb as Decoder_pix2pix
from .Discriminator import Discriminator_asmb as Discriminator_pix2pix

from .Encoder_mnist import Encoder_asmb as Encoder_mnist
from .Decoder_mnist import Decoder_asmb as Decoder_mnist
from .Discriminator_mnist import Discriminator_asmb as Discriminator_mnist

modelset = {
    'pix2pix':{
        'encoder': Encoder_pix2pix,
        'decoder': Decoder_pix2pix,
        'discriminator': Discriminator_pix2pix
    },
    'mnist':{
        'encoder': Encoder_mnist,
        'decoder': Decoder_mnist,
        'discriminator': Discriminator_mnist
    }
}


from .losses import TripletLoss