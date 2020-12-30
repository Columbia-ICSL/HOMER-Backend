from keras.models import model_from_json

fps=10
inputs_path='/Volumes/HUGUETTE/Datasets/Experiments2/'+str(fps)+'_fps'

class Export(object):
    frame_export=True
    no_face_detected_export=False
    video_frame_export_path = inputs_path + '/video_frames'
    video_export_path = inputs_path + '/videos'
    emotions_pred_results_csv_file = inputs_path + '/video_preds_and_highest_emo/data.csv'
    

class Import(object):
    face_detector_file='../Models/Face/haarcascade_frontalface_default.xml'
    video_import_path=inputs_path + '/videos'


class Models(object):
    emotion_model_name='../Models/Emotion/piyush2896_model'
    
    with open(emotion_model_name + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(emotion_model_name + ".h5")
    print("Model loaded!")

    
FD_params= {
                'minDetectRatio': 0.3,
                'scaleFactor': 1.2,
                'minNeighbors': 3,
                'model': 'dlib' #['haarcascade', 'dlib', 'both']

           }