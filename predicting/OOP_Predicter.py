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
from utils.visualiser import Visualiser

class PredictionProducer:
    """
    The purpose of this class is to simplify the process of making a prediction of a test image.
    """
    general_settings = general_settings_obj

    def __init__(self, img_rel_path, saved_trained_model_name):
        # self.general_settings = general_settings_obj
        self.saved_trained_model_name = saved_trained_model_name
        self.img_full_rel_path = self.general_settings["img_path"] + img_rel_path
        self.saved_trained_model_full_path = self.general_settings["saved_model_path"]\
                                             + "/" + self.saved_trained_model_name
        self.model = load_model(self.saved_trained_model_full_path)
        self.prob_type = self.general_settings["classification_problem_type"]["categorical"]
        # self.vis_tool_obj = Visualiser(

    def img_transformer(self):
        img = image.load_img((self.img_full_rel_path),
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

    def get_preds_float_list(self):
        res_dict = {}
        preds_array = self.make_prediction()
        rounded_preds_array = np.around(preds_array, decimals=2)
        flattened_preds_array = rounded_preds_array.flatten()
        pred_probs_list = flattened_preds_array.tolist()
        class_label_names_list = list(self.get_class_label_names_with_pred_probs_dict().keys())
        for class_label_name, class_pred_prob in zip(class_label_names_list, pred_probs_list):
            res_dict[class_label_name] = round(float(class_pred_prob), 2)
        return res_dict


    def display_all_pred_prob_pcts(self):
        trailing_symbol = "|"
        prob_pct = self.make_prediction()
        for pred_label_idx in range(len(self.prob_type["class_label_names"])):
            pred_prob = str(round(prob_pct[0][pred_label_idx], 1))
            output_msg = "{0}: {1} %".format(self.prob_type["class_label_names"][pred_label_idx],
                                     pred_prob)
            print(output_msg, " "*(57-len(output_msg)), trailing_symbol)

    def get_predicted_class_label(self):
        predicted_class_label_idx = self.model.predict_classes(self.img_transformer())
        predicted_class_label_idx = int(predicted_class_label_idx)
        pred_label_name = self.prob_type["class_label_names"][predicted_class_label_idx]
        return pred_label_name

    def get_true_crack_class_label(self):
        all_crack_type_class_labels = self.prob_type["class_label_names"]
        return list(set(all_crack_type_class_labels).intersection(self.img_full_rel_path.split("/")))[0]

    def pred_result_display_msg(self):
        trailing_symbol = "|"
        predicted_class_label_idx = self.model.predict_classes(self.img_transformer())
        predicted_class_label_idx = int(predicted_class_label_idx)
        pred_label_name = self.prob_type["class_label_names"][predicted_class_label_idx]
        prob_pct = self.make_prediction()
        pred_prob = str(round(prob_pct[0][predicted_class_label_idx], 1))
        pred_label_msg = "Type of crack present predicted: {0}".format(pred_label_name)
        pred_pct_msg = "Prediction probability[%]: {0} %".format(pred_prob)
        all_preds_header_msg = "Probabilities of all crack type predictions:"
        true_crack_class_label_msg = "The true crack type class label is: {0}".format(self.get_true_crack_class_label())
        pred_res_msg = "Crack type class label prediction result: {0}".format(self.get_true_crack_class_label() == pred_label_name)
        # THIS IS IT DUDE: print(msg, " "*(50-len(msg2)), "*")
        print("Prediction result message:")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |")
        print(true_crack_class_label_msg, " "*(57-len(true_crack_class_label_msg)), trailing_symbol)
        print(pred_label_msg, " "*(57-len(pred_label_msg)), trailing_symbol)
        print(pred_label_msg, " "*(57-len(pred_label_msg)), trailing_symbol)
        print(pred_res_msg, " "*(57-len(pred_res_msg)), trailing_symbol)
        print(pred_pct_msg, " "*(57-len(pred_pct_msg)), trailing_symbol)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |")
        print(all_preds_header_msg, " "*(57-len(all_preds_header_msg)), trailing_symbol)
        self.display_all_pred_prob_pcts()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |")


    def get_class_label_names_with_pred_probs_dict(self):
        all_pred_probs_array = self.make_prediction()
        pred_probs_rounded = np.round(all_pred_probs_array, decimals=2)
        pred_probs_list = list(pred_probs_rounded[0])
        pred_class_label_names = self.prob_type["class_label_names"]
        class_label_names_and_pred_probs_dict = dict(zip(pred_class_label_names, pred_probs_list))
        return class_label_names_and_pred_probs_dict


    # @property
    def vis_tool_obj(self):
        class_label_names_and_pred_probs_dict = self.get_preds_float_list()
        pred_class_label = self.get_predicted_class_label()
        true_class_label = self.get_true_crack_class_label()
        # return Visualiser(self.img_full_rel_path, pred_class_label,
        #            true_class_label, class_label_names_and_pred_probs_dict).display_pred_res_plot()
        return Visualiser(self.img_full_rel_path, pred_class_label,
                   true_class_label, class_label_names_and_pred_probs_dict)

    def display_pred_res_plot_img(self):
        vis_tool_obj = self.vis_tool_obj()
        vis_tool_obj.display_pred_res_plot()


if __name__ == "__main__":
    # img_path = 'Data/Test/Medium/Crack__20180419_06_19_59,025.bmp'
    img_path = 'Data/Test/Medium/Crack__20180419_06_16_35,563.bmp' # pred : medium and true : medium
    # img_path = 'Data/Test/Medium/Crack__20180419_06_19_09,915.bmp' # pred : small and true : medium
    saved_trained_model_name = "TrainedModel_elu.h5"
    pred = PredictionProducer(img_path, saved_trained_model_name)
    pred.pred_result_display_msg()
    pred.display_pred_res_plot_img()
