import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from configs_and_settings.settings import general_settings_obj
# from predicting.OOP_Predicter import PredictionProducer



class Visualiser:

    def __init__(self, img_rel_path, pred_class_label, true_class_label, pred_res_dict):
        self.general_settings = general_settings_obj
        # self.img_full_rel_path = self.general_settings["img_path"] + img_rel_path
        self.img_full_rel_path = img_rel_path
        self.pred_class_label = pred_class_label
        self.true_class_label = true_class_label
        self.pred_res_dict = pred_res_dict
        self.all_pred_probs_list = list(pred_res_dict.values())
        self.prob_type = self.general_settings["classification_problem_type"]["categorical"]
        self.image = cv2.imread(self.img_full_rel_path)
        self.plot_text = "Large Crack: {Large}%\n" \
                         "Medium Crack: {Medium}%\n" \
                         "Small Crack: {Small}%\n" \
                         "No Crack: {No}%".format(**self.pred_res_dict)

    def display_pred_res_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.5)
        ax.set_title("""\nTrue Crack Classification Label: {0}
        \nPredicted Crack Classification Label: {1}""".format(self.true_class_label,
                                                              self.pred_class_label))
        ax.text(0.95, 0.01, self.plot_text,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='green', fontsize=15)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.show()

# if __name__ == "__main__":
    # '../Data/Test/Medium/Crack__20180419_06_19_59,025.bmp'
    # img_rel_path = '../Data/Test/Medium/Crack__20180419_06_19_59,025.bmp'
    # saved_trained_model_name = "TrainedModel_elu.h5"
    # pred = PredictionProducer(img_rel_path, saved_trained_model_name)
    # img_path = 'Data/Test/Medium/Crack__20180419_06_16_35,563.bmp' # pred : medium and true : medium
    # saved_trained_model_name = "TrainedModel_elu.h5"
    # pred = PredictionProducer(img_path, saved_trained_model_name)
    # vis = pred.vis_tool_obj()
