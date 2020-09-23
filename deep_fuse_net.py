# Style Transfer Network
# Encoder -> AdaIN -> Decoder

import tensorflow as tf

from deepfuse_2.encoder import Encoder
from deepfuse_2.decoder import Decoder
from deepfuse_2.fusion_addition import Strategy
from deepfuse_2.fusion_l1norm import L1_norm

class DeepFuseNet(object):

    def __init__(self, model_pre_path):
        self.encoder = Encoder(model_pre_path)
        self.decoder = Decoder(model_pre_path)

    def transform_addition(self, img1, img2):
        enc_1 = self.encoder.encode(img1)
        enc_2 = self.encoder.encode(img2)
        dec_1 = self.decoder.decode(enc_1)
        dec_2 = self.decoder.decode(enc_2)
        generated_img = Strategy(dec_1, dec_2)
        self.target_feature = generated_img
        return generated_img

    def transform_l1norm(self, img1, img2):
        # encode image
        enc_1 = self.encoder.encode(img1)
        enc_2 = self.encoder.encode(img2)
        dec_1 = self.decoder.decode(enc_1)
        dec_2 = self.decoder.decode(enc_2)
        generated_img = L1_norm(dec_1, dec_2)
        self.target_feature = generated_img
        return generated_img

    def transform_recons(self, img):
        # encode image
        enc = self.encoder.encode(img)
        target_features = enc
        self.target_features = target_features
        generated_img = self.decoder.decode(target_features)
        return generated_img


    def transform_encoder(self, img):
        # encode image
        enc = self.encoder.encode(img)
        return enc

    def transform_decoder(self, feature):
        # decode image
        generated_img = self.decoder.decode(feature)
        return generated_img

