import numpy as np
from settings import Models


'''
class Model(object):
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
    
            # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.summary()
'''
class EmoDetector(object):

    EMOTIONS_LIST7 = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]
    #EMOTIONS_LIST7 = ["Neutral", "Negative", "Positive"]


    def __init__(self):
        # load model from JSON file
        
        self.preds7 = []
        self.best_guess7 = []
        

    def predict_emotion(self, img):
        preds = Models.model.predict(img)
        self.preds7.append(np.squeeze(preds))
        self.best_guess7.append(self.EMOTIONS_LIST7[np.argmax(self.preds7[-1])])
        


    def reinitialization(self):
        self.pred7 = self.best_guess7 = []


if __name__ == '__main__':
    pass