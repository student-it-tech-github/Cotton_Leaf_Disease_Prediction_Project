#Import necessary libraries
from flask import Flask, render_template, request
 
import numpy as np
import os
from waitress import serve
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

#load model
model =load_model("cnn_model_predictor.h5")

print('@@ Model loaded')
def pred_cnn_cld(cld):
  test_image = load_img(cld, target_size = (224, 224)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict class horse or human
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value
  
    
  if pred == 0:
    return('Bacterial Blight')
  elif pred == 1:
    return('Curl Virus Disease')
  elif pred == 2:
    return('Fusarium Wilt Disease')
  elif pred == 3:
    return('Healthy Cotton Leaf')
 

   # if pred == 0:
  #      return "bacterial blight" # if index 0 
  #  elif pred == 1:
#        return "curl virus" # if index 1
 #   elif pred == 2:
  #      return "fusarium wilt" # if index 1
   # elif pred == 3:
    #    return "Healthy" # if index 1
 
#------------>>pred_human_horse<<--end

# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
   
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user upload', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred = pred_cnn_cld(cld=file_path)
               
        return render_template('predict.html', pred_output = pred, user_image = file_path)
    
    #Fo local system
if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=8080)
   # app.run(threaded=False,) 
   
 
 

