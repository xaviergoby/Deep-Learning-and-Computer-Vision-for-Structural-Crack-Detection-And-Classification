# -*- coding: utf-8 -*-
"""
Created on 09/10/2018

@author: Xavier O'Rourke Goby
"""
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from configs_and_settings.settings import general_settings_obj

class PredictionProducer:
    """
    The purpose of this class is to simplify the process of making a prediction of a test image.
    """
    general_settings = general_settings_obj

    def __init__(self, img_rel_path, saved_trained_model_name):
        # self.general_settings = general_settings_obj
        self.saved_trained_model_name = saved_trained_model_name
        self.img_full_rel_pth = self.general_settings["img_path"] + img_rel_path
        self.saved_trained_model_full_path = self.general_settings["saved_model_path"]\
                                             + "/" + self.saved_trained_model_name
        self.model = load_model(self.saved_trained_model_full_path)
        self.prob_type = self.general_settings["classification_problem_type"]["categorical"]

    def img_transformer(self):
        img = image.load_img((self.img_full_rel_pth),
                             target_size=(self.general_settings["img_width"],
                                          self.general_settings["img_height"]))
        img_array = image.img_to_array(img)
        img_array = img_array / 255
        img_array = np.expand_dims(img_array, axis = 0)
        ready_img_array = np.vstack([img_array])
        return ready_img_array

    def make_prediction(self):
        class_probs = self.model.predict(self.img_transformer())
        class_probs_pct = class_probs * 100
        return class_probs_pct

    def display_all_pred_prob_pcts(self):
        prob_pct = self.make_prediction()
        for pred_label_idx in range(len(self.prob_type["class_label_names"])):
            pred_prob = str(round(prob_pct[0][pred_label_idx], 1))
            print("{0}: {1}%".format(self.prob_type["class_label_names"][pred_label_idx],
                                     pred_prob))

    def pred_result_display_msg(self):
        predicted_class_label_idx = self.model.predict_classes(self.img_transformer())
        # prob_type = self.general_settings["classification_problem_type"]["categorical"]
        predicted_class_label_idx = int(predicted_class_label_idx)
        pred_label_name = self.prob_type["class_label_names"][predicted_class_label_idx]
        prob_pct = self.make_prediction()
        pred_prob = str(round(prob_pct[0][predicted_class_label_idx], 1))
        pred_pct_msg = "Predicted a {0} with a probability of {1}%".format(pred_label_name, pred_prob)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(pred_pct_msg)
        print("\nPercentage probabilities of all class labels:")
        self.display_all_pred_prob_pcts()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")









if __name__ == "__main__":
    img_path = 'Data/Test/Medium/Crack__20180419_06_19_59,025.bmp'
    saved_trained_model_name = "TrainedModel_elu.h5"
    pred = PredictionProducer(img_path, saved_trained_model_name)
    # print(pred.img_full_rel_pth)
    # print(pred.saved_trained_model_full_path)
    pred.pred_result_display_msg()